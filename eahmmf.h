// eahmmf.h - Expert-level memory placement (SSD / RAM / VRAM) for the qwen3moe engine.
//
//   Router -> expert ids -> ExpertDescriptor -> ExpertStore -> ExpertPager -> WeightView -> matmul()
//
// Design rules (see EAHMMF requirements, sections 4.x):
//  * An expert is identified by (layer, expert_id) and described by *file offsets* into the GGUF,
//    never by raw mmap addresses. Pointers into mmap/RAM/VRAM are resolved at acquire time.
//  * The GGUF file (mmap) is the immutable SSD/cold tier and is the backing copy of every expert.
//  * Residency is exclusive between RAM and VRAM: a VRAM-resident expert has no RAM copy.
//  * All state changes go through ExpertStore (explicit transition table); ExpertPager only moves
//    bytes and reports success/failure. Migration = copy -> validate -> publish -> release old copy.
//  * A pinned expert (pin_count > 0) is never migrated or evicted.
//  * Everything is synchronous (blocking acquire); one mutex serialises store operations.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

namespace eahmmf {

enum class Residency : uint8_t { SSD = 0, RAM = 1, VRAM = 2, LOADING = 3, EVICTING = 4, ERROR = 5 };
inline const char* res_name(Residency r) {
    static const char* n[] = {"SSD", "RAM", "VRAM", "LOADING", "EVICTING", "ERROR"};
    return n[(int)r];
}
enum Comp { GATE = 0, UP = 1, DOWN = 2 };

// A whole stacked GGUF expert tensor, e.g. ffn_gate_exps.weight with ne = [ne0, ne1, ne2=n_expert].
struct TensorRef {
    uint64_t file_offset = 0;  // absolute offset of the tensor data inside the GGUF file
    uint32_t type = 0;         // ggml type id
    int64_t ne0 = 0, ne1 = 0, ne2 = 0;
    size_t row_bytes = 0;
};
// One expert's component inside the GGUF file. Position-independent (offset, not pointer).
struct TensorSlice {
    uint64_t file_offset = 0;
    size_t bytes = 0;
    uint32_t type = 0;
    int64_t ne0 = 0, ne1 = 0;  // row length, number of rows
    size_t row_bytes = 0;
};

// Everything the matmul kernel needs. The kernel does not know where the bytes live.
struct WeightView {
    const uint8_t* dev = nullptr;  // VRAM address of this component (GPU kernels only; null otherwise)
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    uint32_t type = 0;
    int64_t ne0 = 0, ne1 = 0;
    size_t row_bytes = 0;
    Residency origin = Residency::SSD;  // informational only
};

inline uint64_t hash64(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
        h ^= h >> 29;
    }
    for (; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
static inline bool take_fault(int& n) {
    if (n > 0) { n--; return true; }
    return false;
}
using Clock = std::chrono::steady_clock;
inline double us_since(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }

// ---------------------------------------------------------------------------------------------
// Device (VRAM) backend interface.
// ---------------------------------------------------------------------------------------------
struct DeviceBackend {
    virtual ~DeviceBackend() {}
    virtual const char* name() const = 0;
    virtual void* alloc(size_t bytes) = 0;  // nullptr on failure
    virtual void release(void* p, size_t bytes) = 0;
    virtual bool h2d(void* dev_dst, const void* host_src, size_t n) = 0;
    virtual bool d2h(void* host_dst, const void* dev_src, size_t n) = 0;
    virtual bool sync() = 0;                // copies are valid for compute only after sync()
    virtual bool cpu_readable() const = 0;  // can CPU kernels dereference device pointers?
    virtual size_t total_memory() const { return SIZE_MAX; }
    // Current physical free memory; SIZE_MAX means the backend has no physical limit.
    virtual size_t free_memory() const { return SIZE_MAX; }
};

// Simulated device: separate allocations, poisoned on alloc/free so use of unpublished or freed
// "VRAM" shows up as garbage in tests. Supports fault injection.
struct SimDevice : DeviceBackend {
    int fail_alloc = 0, fail_h2d = 0, partial_h2d = 0, corrupt_h2d = 0, fail_d2h = 0;
    uint64_t n_alloc = 0, n_free = 0, n_h2d = 0, n_d2h = 0, bytes_h2d = 0, bytes_d2h = 0;
    size_t live_bytes = 0, pending = 0;
    const char* name() const override { return "sim"; }
    void* alloc(size_t n) override {
        if (take_fault(fail_alloc)) return nullptr;
        void* p = nullptr;
        if (posix_memalign(&p, 4096, n)) return nullptr;
        memset(p, 0xCD, n);
        n_alloc++;
        live_bytes += n;
        return p;
    }
    void release(void* p, size_t n) override {
        if (!p) return;
        memset(p, 0xDD, n);
        free(p);
        n_free++;
        live_bytes -= n;
    }
    bool h2d(void* dst, const void* src, size_t n) override {
        if (take_fault(fail_h2d)) return false;
        if (take_fault(partial_h2d)) { memcpy(dst, src, n / 2); return false; }
        memcpy(dst, src, n);
        if (take_fault(corrupt_h2d)) ((uint8_t*)dst)[n / 3] ^= 0x5A;
        pending += n;
        n_h2d++;
        bytes_h2d += n;
        return true;
    }
    bool d2h(void* dst, const void* src, size_t n) override {
        if (take_fault(fail_d2h)) return false;
        memcpy(dst, src, n);
        n_d2h++;
        bytes_d2h += n;
        return true;
    }
    bool sync() override { pending = 0; return true; }
    bool cpu_readable() const override { return true; }
};

#ifdef EAHMMF_CUDA
}  // namespace eahmmf
#include <cuda_runtime.h>
namespace eahmmf {
// Real GPU memory. CPU kernels cannot dereference it, so ExpertStore bounces a pinned expert back
// to a host scratch buffer for the (CPU) matmul. A device matmul kernel would consume the device
// pointer directly instead. NOTE: not compiled/tested in the development sandbox (no GPU there).
struct CudaDevice : DeviceBackend {
    cudaStream_t st{};
    size_t total = 0;
    CudaDevice() {
        cudaStreamCreate(&st);
        size_t fr = 0;
        cudaMemGetInfo(&fr, &total);
        total = fr;
    }
    ~CudaDevice() override { cudaStreamDestroy(st); }
    const char* name() const override { return "cuda"; }
    void* alloc(size_t n) override {
        void* p = nullptr;
        cudaError_t err = cudaMalloc(&p, n);
        if (err != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);

            fprintf(stderr,
                    "[EAHMMF CUDA] cudaMalloc failed: requested=%zu bytes, "
                    "error=%s (%d), free=%zu MiB, total=%zu MiB",
                    n, cudaGetErrorString(err), (int)err,
                    mem_err == cudaSuccess ? free_bytes / (1024 * 1024) : 0,
                    mem_err == cudaSuccess ? total_bytes / (1024 * 1024) : 0);

            if (mem_err != cudaSuccess)
                fprintf(stderr, ", cudaMemGetInfo error=%s",
                        cudaGetErrorString(mem_err));

            fprintf(stderr, "\n");
            return nullptr;
        }
        return p;
    }
    void release(void* p, size_t) override { if (p) cudaFree(p); }
    bool h2d(void* d, const void* s, size_t n) override { return cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, st) == cudaSuccess; }
    bool d2h(void* d, const void* s, size_t n) override { return cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToHost, st) == cudaSuccess && cudaStreamSynchronize(st) == cudaSuccess; }
    bool sync() override { return cudaStreamSynchronize(st) == cudaSuccess; }
    bool cpu_readable() const override { return false; }
    size_t total_memory() const override { return total; }
    size_t free_memory() const override {
        size_t fr = 0, total_bytes = 0;
        return cudaMemGetInfo(&fr, &total_bytes) == cudaSuccess ? fr : 0;
    }
};
#endif

// ---------------------------------------------------------------------------------------------
// Config, telemetry
// ---------------------------------------------------------------------------------------------
struct Config {
    Residency exec_tier = Residency::SSD;  // tier experts are brought to before execution
    size_t ram_budget = SIZE_MAX;          // bytes available for managed RAM expert copies
    size_t vram_budget = SIZE_MAX;         // bytes available in the VRAM pool
    size_t vram_reserve = 0;               // physical free VRAM reserved for CUDA/runtime overhead
    bool verify = false;                   // checksum every migration against the GGUF source
    bool gpu_compute = false;  // VRAM copies are consumed by GPU kernels directly (no host bounce copy)
    bool strict = false;                   // fail acquire instead of falling back to a lower tier
    bool record_events = false;            // keep events in memory (tests)
    std::string log_path;                  // write the residency event log here
    bool log_acquires = true;              // include per-acquire lines in the log
};

struct Event {
    enum Kind { ACQUIRE, RELEASE, MIGRATE, FALLBACK, REFUSED } kind = ACQUIRE;
    int layer = -1, expert = -1;
    Residency from = Residency::SSD, to = Residency::SSD;
    size_t bytes = 0;
    double lat_us = 0;
    bool ok = true;
    uint32_t pin = 0;
    size_t ram_used = 0, vram_used = 0;
    std::string note;
};
enum MigPath { SSD_RAM = 0, RAM_VRAM = 1, VRAM_RAM = 2, RAM_SSD = 3, VRAM_SSD = 4 };
static const char* kPathName[5] = {"SSD->RAM", "RAM->VRAM", "VRAM->RAM", "RAM->SSD", "VRAM->SSD"};

struct Stats {  // physical residency telemetry (routing telemetry is kept elsewhere)
    uint64_t acquires = 0, releases = 0, pins = 0, unpins = 0, pinned_now = 0, pinned_max = 0;
    uint64_t hit[3] = {0, 0, 0}, miss[3] = {0, 0, 0};  // indexed by Residency SSD/RAM/VRAM
    uint64_t mig_n[5] = {}, mig_fail[5] = {}, mig_bytes[5] = {};
    double mig_us[5] = {};
    uint64_t fallbacks = 0, refused_evictions = 0;
    uint64_t vram_alloc_failures = 0, vram_alloc_retries = 0;
    uint64_t vram_recovery_evictions = 0;
    uint64_t vram_recovered_allocations = 0, vram_unrecovered_allocations = 0;
    double acquire_us = 0, acquire_us_max = 0;
};

struct ExpertDescriptor {
    int layer = -1, expert_id = -1;
    TensorSlice slice[3];            // gate, up, down inside the GGUF
    size_t bytes[3] = {0, 0, 0};     // payload bytes per component
    size_t total_bytes = 0;          // sum of payload bytes
    size_t res_off[3] = {0, 0, 0};   // component offsets inside a resident (RAM/VRAM) buffer
    size_t footprint = 0;            // bytes charged against RAM/VRAM budgets (page aligned)
    Residency state = Residency::SSD, prev_state = Residency::SSD;
    uint8_t ram_mask = 0, vram_mask = 0, pending_mask = 0;  // components present (bit per Comp)
    void* ram_ptr = nullptr;
    void* vram_ptr = nullptr;
    void* bounce_ptr = nullptr;      // host scratch for non-CPU-readable devices
    uint32_t pin_count = 0;
    uint64_t last_use = 0;
    uint64_t src_hash = 0;
    bool src_hash_valid = false;
    bool fully_resident_ram() const { return ram_mask == 7; }
    bool fully_resident_vram() const { return vram_mask == 7; }
    bool partially_resident() const { return pending_mask != 0 && pending_mask != 7; }
};

struct ExpertViews {
    WeightView gate, up, down;
    bool ok = false;
    Residency served_from = Residency::SSD;
};

class ExpertStore;

// Owns exactly one successful acquire() pin.
// The ExpertStore must outlive every lease acquired from it.
class ExpertLease {
public:
    ExpertLease() = default;
    ExpertLease(const ExpertLease&) = delete;
    ExpertLease& operator=(const ExpertLease&) = delete;

    ExpertLease(ExpertLease&& other) noexcept;
    ExpertLease& operator=(ExpertLease&& other) noexcept;

    ~ExpertLease();

    explicit operator bool() const noexcept { return store_ != nullptr; }
    const ExpertViews& views() const noexcept { return views_; }
    void reset() noexcept;

private:
    friend class ExpertStore;

    ExpertLease(ExpertStore* store, int layer, int expert,
                const ExpertViews& views) noexcept;

    ExpertStore* store_ = nullptr;
    int layer_ = -1;
    int expert_ = -1;
    ExpertViews views_;
};

class ExpertPager;

// ---------------------------------------------------------------------------------------------
// ExpertStore: logical->physical mapping, capacity accounting, pinning, state transitions.
// ---------------------------------------------------------------------------------------------
class ExpertStore {
public:
    ExpertStore(const uint8_t* base, size_t size, int n_layer, int n_expert, const std::vector<std::array<TensorRef, 3>>& refs,
                const Config& cfg, DeviceBackend* dev);
    ~ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    // --- required operations ---
    ExpertViews acquire(int layer, int expert);  // stage per policy, pin, return views (ok=false on failure)
    ExpertLease acquire_lease(int layer, int expert); // RAII pin ownership
    bool release(int layer, int expert);         // unpin
    Residency residency(int layer, int expert);
    // --- explicit placement (tests / future policy layer); no pinning ---
    bool promote(int layer, int expert, Residency tier);  // move up to `tier` (RAM or VRAM)
    bool demote(int layer, int expert, Residency tier);   // move down to `tier` (RAM or SSD)
    // --- introspection ---
    ExpertDescriptor descriptor(int layer, int expert) const {
        std::lock_guard<std::recursive_mutex> g(mu_);
        return descs_.at(idx(layer, expert));
    }
    std::string where(int layer, int expert);
    size_t ram_total() const { std::lock_guard<std::recursive_mutex> g(mu_); return ram_cap_; }
    size_t ram_used() const { std::lock_guard<std::recursive_mutex> g(mu_); return ram_used_; }
    size_t vram_total() const { std::lock_guard<std::recursive_mutex> g(mu_); return vram_cap_; }
    size_t vram_used() const { std::lock_guard<std::recursive_mutex> g(mu_); return vram_used_; }
    size_t count_in(Residency r);
    Stats stats() const {
        std::lock_guard<std::recursive_mutex> g(mu_);
        return st_;
    }
    std::vector<Event> events() const {
        std::lock_guard<std::recursive_mutex> g(mu_);
        return events_;
    }
    std::string check_invariants();  // "" if consistent
    void print_summary(FILE* f);
    Config config() const {
        std::lock_guard<std::recursive_mutex> g(mu_);
        return cfg_;
    }
    void set_strict(bool enabled) {
        std::lock_guard<std::recursive_mutex> g(mu_);
        cfg_.strict = enabled;
    }
    // Test-only fault injection; all counters are updated atomically.
    void set_fault_injection(int backing_copy, int ram_alloc, int corrupt_copy) {
        std::lock_guard<std::recursive_mutex> g(mu_);
        fail_backing_copy = backing_copy;
        fail_ram_alloc = ram_alloc;
        corrupt_backing_copy = corrupt_copy;
    }

private:
    friend class ExpertPager;
    size_t idx(int l, int e) const { return (size_t)l * n_expert_ + e; }
    ExpertDescriptor& at(int l, int e) { return descs_.at(idx(l, e)); }
    // transition machinery
    static bool allowed(Residency from, Residency to);
    void begin(ExpertDescriptor& e, Residency to);
    void abort_transition(ExpertDescriptor& e);
    void set_state(ExpertDescriptor& e, Residency to);
    // resources
    void* ram_alloc(size_t fp);
    void ram_free(ExpertDescriptor& e);
    void vram_free(ExpertDescriptor& e);
    bool ensure_ram_room(size_t need, const ExpertDescriptor* exclude);
    bool ensure_vram_room(size_t need, const ExpertDescriptor* exclude);
    ExpertDescriptor* lru_victim(Residency tier, const ExpertDescriptor* exclude);
    // data helpers
    const uint8_t* backing(const ExpertDescriptor& e, int c) const { return base_ + e.slice[c].file_offset; }
    uint64_t source_hash(ExpertDescriptor& e);
    uint64_t resident_hash(const ExpertDescriptor& e, const uint8_t* buf) const;
    uint64_t device_hash(const ExpertDescriptor& e, const void* dev_ptr);
    WeightView make_view(const ExpertDescriptor& e, int c, const uint8_t* buf, Residency origin) const;
    bool pin_bounce(ExpertDescriptor& e);
    // telemetry
    void emit(const Event& ev);
    void log_migration(ExpertDescriptor& e, MigPath p, Residency from, Residency to, size_t bytes, double lat, bool ok, const char* note);

    const uint8_t* base_;
    size_t size_;
    int n_layer_, n_expert_;
    Config cfg_;
    DeviceBackend* dev_;
    std::vector<ExpertDescriptor> descs_;
    size_t ram_cap_, vram_cap_, ram_used_ = 0, vram_used_ = 0;
    uint64_t tick_ = 0;
    Stats st_;
    std::vector<Event> events_;
    FILE* log_ = nullptr;
    Clock::time_point t_start_ = Clock::now();
    mutable std::recursive_mutex mu_;
    std::unique_ptr<ExpertPager> pager_;
    // Test-only fault injection counters; accessed under mu_.
    int fail_backing_copy = 0, fail_ram_alloc = 0, corrupt_backing_copy = 0;
};

// ---------------------------------------------------------------------------------------------
// ExpertPager: moves expert bytes between tiers. Every path is transactional:
//   begin(LOADING/EVICTING) -> allocate -> copy -> validate -> publish (via store) -> release old copy.
// On any failure the destination is freed and the previous valid residency is restored.
// ---------------------------------------------------------------------------------------------
class ExpertPager {
public:
    explicit ExpertPager(ExpertStore& s) : S(s) {}

private:
    friend class ExpertStore;

    bool stage_to_ram(ExpertDescriptor& e);       // SSD  -> RAM
    bool stage_to_vram(ExpertDescriptor& e);      // RAM  -> VRAM
    bool evict_vram_to_ram(ExpertDescriptor& e);  // VRAM -> RAM
    bool evict_ram_to_ssd(ExpertDescriptor& e);   // RAM  -> SSD (drop RAM copy)
    bool drop_vram(ExpertDescriptor& e);          // VRAM -> SSD (fallback when RAM cannot take the copy)

    ExpertStore& S;
};

// =============================================================================================
// ExpertStore implementation
// =============================================================================================
inline ExpertStore::ExpertStore(const uint8_t* base, size_t size, int n_layer, int n_expert,
                                const std::vector<std::array<TensorRef, 3>>& refs, const Config& cfg, DeviceBackend* dev)
    : base_(base), size_(size), n_layer_(n_layer), n_expert_(n_expert), cfg_(cfg), dev_(dev) {
    ram_cap_ = cfg.ram_budget;
    vram_cap_ = dev ? std::min(cfg.vram_budget, dev->total_memory()) : 0;
    descs_.resize((size_t)n_layer * n_expert);
    for (int l = 0; l < n_layer; l++)
        for (int x = 0; x < n_expert; x++) {
            ExpertDescriptor& d = at(l, x);
            d.layer = l;
            d.expert_id = x;
            size_t off = 0;
            for (int c = 0; c < 3; c++) {
                const TensorRef& r = refs[l][c];
                TensorSlice& s = d.slice[c];
                s.bytes = (size_t)r.ne1 * r.row_bytes;
                s.file_offset = r.file_offset + (uint64_t)x * s.bytes;
                s.type = r.type;
                s.ne0 = r.ne0;
                s.ne1 = r.ne1;
                s.row_bytes = r.row_bytes;
                d.bytes[c] = s.bytes;
                d.total_bytes += s.bytes;
                d.res_off[c] = off;
                off = align_up(off + s.bytes, 64);
                if (x >= r.ne2 || s.file_offset + s.bytes > size) d.state = Residency::ERROR;  // backing range unavailable
            }
            d.footprint = align_up(off, 4096);
            d.prev_state = d.state;
        }
    if (!cfg.log_path.empty()) {
        log_ = fopen(cfg.log_path.c_str(), "w");
        if (!log_) fprintf(stderr, "warning: cannot open expert log %s\n", cfg.log_path.c_str());
    }
    pager_.reset(new ExpertPager(*this));
}
inline ExpertStore::~ExpertStore() {
    // A live lease still owns a pin. Never free buffers while leases exist.
    for (const auto& d : descs_) {
        if (d.pin_count != 0) {
            fprintf(stderr,
                    "eahmmf: ExpertStore destroyed with %u outstanding pin(s)\n",
                    static_cast<unsigned>(d.pin_count));
            std::abort();
        }
    }

    for (auto& d : descs_) {
        if (d.bounce_ptr) { free(d.bounce_ptr); d.bounce_ptr = nullptr; }
        if (d.ram_ptr) { free(d.ram_ptr); d.ram_ptr = nullptr; }
        if (d.vram_ptr && dev_) { dev_->release(d.vram_ptr, d.footprint); d.vram_ptr = nullptr; }
    }
    if (log_) fclose(log_);
}

inline bool ExpertStore::allowed(Residency from, Residency to) {
    using R = Residency;
    if (to == R::ERROR) return true;
    if (from == R::ERROR) return to == R::SSD;
    bool stable = from == R::SSD || from == R::RAM || from == R::VRAM;
    if (stable) return to == R::LOADING || to == R::EVICTING;
    return to == R::SSD || to == R::RAM || to == R::VRAM;  // LOADING / EVICTING settle into a stable state
}
inline void ExpertStore::set_state(ExpertDescriptor& e, Residency to) {
    if (!allowed(e.state, to)) {
        fprintf(stderr, "eahmmf: illegal transition %s -> %s for (layer %d, expert %d)\n", res_name(e.state), res_name(to), e.layer, e.expert_id);
        abort();
    }
    e.state = to;
}
inline void ExpertStore::begin(ExpertDescriptor& e, Residency to) {
    e.prev_state = e.state;
    set_state(e, to);
    e.pending_mask = 0;
}
inline void ExpertStore::abort_transition(ExpertDescriptor& e) {
    set_state(e, e.prev_state);  // the previous residency was never invalidated
    e.pending_mask = 0;
}

inline void* ExpertStore::ram_alloc(size_t fp) {
    if (take_fault(fail_ram_alloc)) return nullptr;
    void* p = nullptr;
    if (posix_memalign(&p, 4096, fp)) return nullptr;
    memset(p, 0, fp);
    return p;
}
inline void ExpertStore::ram_free(ExpertDescriptor& e) {
    if (!e.ram_ptr) return;
    free(e.ram_ptr);
    e.ram_ptr = nullptr;
    e.ram_mask = 0;
    ram_used_ -= e.footprint;
}
inline void ExpertStore::vram_free(ExpertDescriptor& e) {
    if (!e.vram_ptr) return;
    dev_->release(e.vram_ptr, e.footprint);
    e.vram_ptr = nullptr;
    e.vram_mask = 0;
    vram_used_ -= e.footprint;
}

inline ExpertDescriptor* ExpertStore::lru_victim(Residency tier, const ExpertDescriptor* exclude) {
    ExpertDescriptor* best = nullptr;
    for (auto& d : descs_) {
        if (d.state != tier || d.pin_count || &d == exclude) continue;
        if (!best || d.last_use < best->last_use) best = &d;
    }
    return best;
}
inline bool ExpertStore::ensure_ram_room(size_t need, const ExpertDescriptor* exclude) {
    if (need > ram_cap_) return false;
    while (ram_used_ + need > ram_cap_) {
        ExpertDescriptor* v = lru_victim(Residency::RAM, exclude);
        if (!v || !pager_->evict_ram_to_ssd(*v)) return false;
    }
    return true;
}
inline bool ExpertStore::ensure_vram_room(size_t need, const ExpertDescriptor* exclude) {
    if (!dev_ || need > vram_cap_) return false;
    for (;;) {
        const size_t free = dev_->free_memory();
        const bool logical_pressure = vram_used_ > vram_cap_ - need;
        const bool physical_pressure = free < need || free - need < cfg_.vram_reserve;
        if (!logical_pressure && !physical_pressure) break;

        ExpertDescriptor* v = lru_victim(Residency::VRAM, exclude);
        if (!v) { st_.refused_evictions++; return false; }  // no unpinned resident can free enough memory
        if (!pager_->evict_vram_to_ram(*v) && !pager_->drop_vram(*v)) return false;
    }
    return true;
}

inline uint64_t ExpertStore::resident_hash(const ExpertDescriptor& e, const uint8_t* buf) const {
    uint64_t h = 0;
    for (int c = 0; c < 3; c++) h = h * 1000003ULL + hash64(buf + e.res_off[c], e.bytes[c]);
    return h;
}
inline uint64_t ExpertStore::source_hash(ExpertDescriptor& e) {
    if (!e.src_hash_valid) {
        uint64_t h = 0;
        for (int c = 0; c < 3; c++) h = h * 1000003ULL + hash64(backing(e, c), e.bytes[c]);
        e.src_hash = h;
        e.src_hash_valid = true;
    }
    return e.src_hash;
}
inline uint64_t ExpertStore::device_hash(const ExpertDescriptor& e, const void* dp) {
    if (dev_->cpu_readable()) return resident_hash(e, (const uint8_t*)dp);
    std::vector<uint8_t> tmp(e.footprint);
    for (int c = 0; c < 3; c++)
        if (!dev_->d2h(tmp.data() + e.res_off[c], (const uint8_t*)dp + e.res_off[c], e.bytes[c])) return ~0ULL;
    return resident_hash(e, tmp.data());
}

inline WeightView ExpertStore::make_view(const ExpertDescriptor& e, int c, const uint8_t* buf, Residency origin) const {
    WeightView v;
    const TensorSlice& s = e.slice[c];
    v.data = buf ? buf + e.res_off[c] : base_ + s.file_offset;
    v.bytes = s.bytes;
    v.type = s.type;
    v.ne0 = s.ne0;
    v.ne1 = s.ne1;
    v.row_bytes = s.row_bytes;
    v.origin = origin;
    return v;
}

// Non-CPU-readable devices: give the CPU kernel a host copy for the duration of the pin.
inline bool ExpertStore::pin_bounce(ExpertDescriptor& e) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, e.footprint)) return false;
    for (int c = 0; c < 3; c++)
        if (!dev_->d2h((uint8_t*)p + e.res_off[c], (uint8_t*)e.vram_ptr + e.res_off[c], e.bytes[c])) { free(p); return false; }
    e.bounce_ptr = p;
    return true;
}

inline void ExpertStore::emit(const Event& ev) {
    if (cfg_.record_events) events_.push_back(ev);
    if (!log_) return;
    double t = std::chrono::duration<double>(Clock::now() - t_start_).count();
    switch (ev.kind) {
        case Event::ACQUIRE:
            if (cfg_.log_acquires)
                fprintf(log_, "%.6f ACQUIRE layer=%d expert=%d served=%s pin=%u lat_us=%.1f %s\n", t, ev.layer, ev.expert, res_name(ev.to), ev.pin, ev.lat_us, ev.note.c_str());
            break;
        case Event::RELEASE:
            if (cfg_.log_acquires) fprintf(log_, "%.6f RELEASE layer=%d expert=%d pin=%u\n", t, ev.layer, ev.expert, ev.pin);
            break;
        case Event::MIGRATE:
            fprintf(log_, "%.6f MIGRATE layer=%d expert=%d %s->%s bytes=%zu lat_us=%.1f ok=%d ram=%zu/%zu vram=%zu/%zu %s\n", t, ev.layer, ev.expert,
                    res_name(ev.from), res_name(ev.to), ev.bytes, ev.lat_us, (int)ev.ok, ev.ram_used, ram_cap_, ev.vram_used, vram_cap_, ev.note.c_str());
            break;
        case Event::FALLBACK:
            fprintf(log_, "%.6f FALLBACK layer=%d expert=%d served=%s %s\n", t, ev.layer, ev.expert, res_name(ev.to), ev.note.c_str());
            break;
        case Event::REFUSED:
            fprintf(log_, "%.6f REFUSED layer=%d expert=%d %s\n", t, ev.layer, ev.expert, ev.note.c_str());
            break;
    }
}
inline void ExpertStore::log_migration(ExpertDescriptor& e, MigPath p, Residency from, Residency to, size_t bytes, double lat, bool ok, const char* note) {
    if (ok) { st_.mig_n[p]++; st_.mig_bytes[p] += bytes; st_.mig_us[p] += lat; }
    else st_.mig_fail[p]++;
    Event ev;
    ev.kind = Event::MIGRATE;
    ev.layer = e.layer;
    ev.expert = e.expert_id;
    ev.from = from;
    ev.to = to;
    ev.bytes = bytes;
    ev.lat_us = lat;
    ev.ok = ok;
    ev.ram_used = ram_used_;
    ev.vram_used = vram_used_;
    ev.note = note;
    emit(ev);
}

inline Residency ExpertStore::residency(int l, int e) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    return at(l, e).state;
}
inline size_t ExpertStore::count_in(Residency r) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    size_t n = 0;
    for (auto& d : descs_) n += d.state == r;
    return n;
}
inline std::string ExpertStore::where(int l, int e) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    const ExpertDescriptor& d = at(l, e);
    char b[256];
    snprintf(b, sizeof b, "(L%d,E%d) state=%s gguf@%llu+%zu ram=%s vram=%s pin=%u", l, e, res_name(d.state), (unsigned long long)d.slice[0].file_offset, d.total_bytes,
             d.ram_ptr ? "yes" : "no", d.vram_ptr ? "yes" : "no", d.pin_count);
    return b;
}

inline ExpertViews ExpertStore::acquire(int l, int x) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    auto t0 = Clock::now();
    ExpertDescriptor& d = at(l, x);
    ExpertViews out;
    st_.acquires++;
    auto finish = [&](const char* note) {
        double lat = us_since(t0);
        st_.acquire_us += lat;
        st_.acquire_us_max = std::max(st_.acquire_us_max, lat);
        if (cfg_.record_events || log_) {
            Event ev;
            ev.kind = Event::ACQUIRE;
            ev.layer = l;
            ev.expert = x;
            ev.to = out.ok ? out.served_from : Residency::ERROR;
            ev.lat_us = lat;
            ev.ok = out.ok;
            ev.pin = d.pin_count;
            ev.ram_used = ram_used_;
            ev.vram_used = vram_used_;
            ev.note = note;
            emit(ev);
        }
    };
    if (d.state == Residency::ERROR) { finish("unavailable"); return out; }
    const Residency want = cfg_.exec_tier;
    const Residency entry = d.state;
    // cascade accounting: VRAM -> RAM -> SSD
    if (want == Residency::VRAM) (entry == Residency::VRAM ? st_.hit : st_.miss)[2]++;
    if (want != Residency::SSD && entry != Residency::VRAM) (entry == Residency::RAM ? st_.hit : st_.miss)[1]++;
    if (entry == Residency::SSD) st_.hit[0]++;

    // Upgrade towards the execution tier (never while pinned: that would release a copy in use).
    if (d.pin_count == 0) {
        bool ok = true;
        if (want != Residency::SSD && d.state == Residency::SSD) ok = pager_->stage_to_ram(d);
        if (ok && want == Residency::VRAM && d.state == Residency::RAM) ok = pager_->stage_to_vram(d);
        if (!ok) {
            st_.fallbacks++;
            Event ev;
            ev.kind = Event::FALLBACK;
            ev.layer = l;
            ev.expert = x;
            ev.to = d.state;
            ev.note = "staging failed";
            emit(ev);
            if (cfg_.strict) { finish("strict: staging failed"); return out; }
        }
    } else if (want == Residency::VRAM && d.state != Residency::VRAM) {
        st_.fallbacks++;  // pinned experts are not migrated: serve the current valid copy
    }
    // Resolve a readable buffer for the *current valid* residency.
    const uint8_t* buf = nullptr;
    Residency from = d.state;
    bool gpu = false;
    if (d.state == Residency::RAM) buf = (const uint8_t*)d.ram_ptr;
    else if (d.state == Residency::VRAM) {
        if (dev_->cpu_readable()) buf = (const uint8_t*)d.vram_ptr;
        else if (cfg_.gpu_compute) gpu = true;  // GPU reads the VRAM copy; host view falls back to the SSD mapping
        else if (d.bounce_ptr || pin_bounce(d)) buf = (const uint8_t*)d.bounce_ptr;
        else {  // could not make the device copy readable: the SSD backing is still valid
            if (cfg_.strict) { finish("strict: bounce failed"); return out; }
            st_.fallbacks++;
            from = Residency::SSD;
        }
    }
    out.gate = make_view(d, GATE, buf, from);
    out.up = make_view(d, UP, buf, from);
    out.down = make_view(d, DOWN, buf, from);
    out.served_from = (buf || gpu) ? from : Residency::SSD;
    if (gpu) {
        const uint8_t* dp = (const uint8_t*)d.vram_ptr;
        out.gate.dev = dp + d.res_off[GATE];
        out.up.dev = dp + d.res_off[UP];
        out.down.dev = dp + d.res_off[DOWN];
    }
    out.ok = true;
    d.pin_count++;
    d.last_use = ++tick_;
    st_.pins++;
    if (d.pin_count == 1) { st_.pinned_now++; st_.pinned_max = std::max(st_.pinned_max, st_.pinned_now); }
    finish(entry == out.served_from ? "hit" : "staged");
    return out;
}

inline bool ExpertStore::release(int l, int x) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    ExpertDescriptor& d = at(l, x);
    if (d.pin_count == 0) {
        fprintf(stderr, "eahmmf: release of unpinned expert (layer %d, expert %d)\n", l, x);
        return false;
    }
    d.pin_count--;
    st_.releases++;
    st_.unpins++;
    if (d.pin_count == 0) {
        st_.pinned_now--;
        if (d.bounce_ptr) { free(d.bounce_ptr); d.bounce_ptr = nullptr; }
    }
    if (cfg_.record_events || log_) {
        Event ev;
        ev.kind = Event::RELEASE;
        ev.layer = l;
        ev.expert = x;
        ev.pin = d.pin_count;
        emit(ev);
    }
    return true;
}

inline ExpertLease::ExpertLease(ExpertStore* store, int layer, int expert,
                                const ExpertViews& views) noexcept
    : store_(store), layer_(layer), expert_(expert), views_(views) {}

inline ExpertLease::~ExpertLease() {
    reset();
}

inline ExpertLease::ExpertLease(ExpertLease&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)),
      layer_(std::exchange(other.layer_, -1)),
      expert_(std::exchange(other.expert_, -1)),
      views_(std::exchange(other.views_, ExpertViews{})) {}

inline ExpertLease& ExpertLease::operator=(ExpertLease&& other) noexcept {
    if (this != &other) {
        reset();
        store_ = std::exchange(other.store_, nullptr);
        layer_ = std::exchange(other.layer_, -1);
        expert_ = std::exchange(other.expert_, -1);
        views_ = std::exchange(other.views_, ExpertViews{});
    }
    return *this;
}

inline void ExpertLease::reset() noexcept {
    ExpertStore* store = std::exchange(store_, nullptr);
    const int layer = std::exchange(layer_, -1);
    const int expert = std::exchange(expert_, -1);
    views_ = ExpertViews{};

    if (store) {
        store->release(layer, expert);
    }
}

inline ExpertLease ExpertStore::acquire_lease(int layer, int expert) {
    ExpertViews acquired = acquire(layer, expert);
    if (!acquired.ok) {
        return {};
    }
    return ExpertLease(this, layer, expert, acquired);
}

inline bool ExpertStore::promote(int l, int x, Residency tier) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    ExpertDescriptor& d = at(l, x);
    if (d.state == Residency::ERROR) return false;
    if (d.pin_count) {
        st_.refused_evictions++;
        Event ev;
        ev.kind = Event::REFUSED;
        ev.layer = l;
        ev.expert = x;
        ev.note = "expert pinned";
        emit(ev);
        return false;
    }
    bool ok = false;
    if (tier == Residency::RAM) {
        ok = d.state == Residency::RAM || d.state == Residency::VRAM || pager_->stage_to_ram(d);
    } else if (tier == Residency::VRAM) {
        if (d.state == Residency::VRAM) ok = true;
        else ok = (d.state != Residency::SSD || pager_->stage_to_ram(d)) && pager_->stage_to_vram(d);
    } else ok = tier == Residency::SSD;
    if (ok) d.last_use = ++tick_;
    return ok;
}
inline bool ExpertStore::demote(int l, int x, Residency tier) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    ExpertDescriptor& d = at(l, x);
    if (d.pin_count) {
        st_.refused_evictions++;
        Event ev;
        ev.kind = Event::REFUSED;
        ev.layer = l;
        ev.expert = x;
        ev.note = "expert pinned";
        emit(ev);
        return false;
    }
    if (tier == Residency::RAM) return d.state == Residency::RAM || (d.state == Residency::VRAM && pager_->evict_vram_to_ram(d));
    if (tier == Residency::SSD) {
        if (d.state == Residency::SSD) return true;
        if (d.state == Residency::VRAM && !pager_->evict_vram_to_ram(d)) return false;
        return d.state == Residency::RAM && pager_->evict_ram_to_ssd(d);
    }
    return false;
}

inline std::string ExpertStore::check_invariants() {
    std::lock_guard<std::recursive_mutex> g(mu_);
    size_t ru = 0, vu = 0, pinned = 0;
    char b[200];
    for (auto& d : descs_) {
        if (d.ram_ptr) ru += d.footprint;
        if (d.vram_ptr) vu += d.footprint;
        if (d.pin_count) pinned++;
        bool ok = true;
        switch (d.state) {
            case Residency::SSD: ok = !d.ram_ptr && !d.vram_ptr && !d.ram_mask && !d.vram_mask; break;
            case Residency::RAM: ok = d.ram_ptr && !d.vram_ptr && d.ram_mask == 7; break;
            case Residency::VRAM: ok = d.vram_ptr && !d.ram_ptr && d.vram_mask == 7; break;
            case Residency::ERROR: ok = d.pin_count == 0; break;
            default: ok = false;  // transient states must not be visible between operations
        }
        if (d.bounce_ptr && !d.pin_count) ok = false;
        if (!ok) {
            snprintf(b, sizeof b, "inconsistent descriptor (L%d,E%d) state=%s ram=%p vram=%p", d.layer, d.expert_id, res_name(d.state), d.ram_ptr, d.vram_ptr);
            return b;
        }
    }
    if (ru != ram_used_) { snprintf(b, sizeof b, "ram accounting %zu != %zu", ru, ram_used_); return b; }
    if (vu != vram_used_) { snprintf(b, sizeof b, "vram accounting %zu != %zu", vu, vram_used_); return b; }
    if (ram_used_ > ram_cap_ || vram_used_ > vram_cap_) return "capacity exceeded";
    if (pinned != st_.pinned_now) return "pinned counter mismatch";
    return "";
}

inline void ExpertStore::print_summary(FILE* f) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    auto mb = [](size_t b) { return b == SIZE_MAX ? -1.0 : b / 1048576.0; };
    fprintf(f, "\n[expert residency] tier=%s device=%s\n", res_name(cfg_.exec_tier), dev_ ? dev_->name() : "none");
    fprintf(f, "  acquires=%llu releases=%llu pinned_now=%llu pinned_max=%llu avg_acquire=%.1fus max=%.1fus\n", (unsigned long long)st_.acquires,
            (unsigned long long)st_.releases, (unsigned long long)st_.pinned_now, (unsigned long long)st_.pinned_max,
            st_.acquires ? st_.acquire_us / st_.acquires : 0.0, st_.acquire_us_max);
    fprintf(f, "  resident: VRAM=%zu RAM=%zu SSD=%zu | RAM %.1f/%.1f MiB  VRAM %.1f/%.1f MiB (-1 = unlimited)\n", count_in(Residency::VRAM), count_in(Residency::RAM),
            count_in(Residency::SSD), ram_used_ / 1048576.0, mb(ram_cap_), vram_used_ / 1048576.0, mb(vram_cap_));
    fprintf(f, "  cache: VRAM hit/miss=%llu/%llu  RAM hit/miss=%llu/%llu  SSD served=%llu  fallbacks=%llu refused_evictions=%llu\n", (unsigned long long)st_.hit[2],
            (unsigned long long)st_.miss[2], (unsigned long long)st_.hit[1], (unsigned long long)st_.miss[1], (unsigned long long)st_.hit[0],
            (unsigned long long)st_.fallbacks, (unsigned long long)st_.refused_evictions);
    for (int p = 0; p < 5; p++)
        if (st_.mig_n[p] || st_.mig_fail[p])
            fprintf(f, "  %-9s n=%llu fail=%llu bytes=%.2f MiB avg_lat=%.1fus\n", kPathName[p], (unsigned long long)st_.mig_n[p], (unsigned long long)st_.mig_fail[p],
                    st_.mig_bytes[p] / 1048576.0, st_.mig_n[p] ? st_.mig_us[p] / st_.mig_n[p] : 0.0);
    fprintf(f, "  VRAM alloc: failures=%llu retries=%llu recovery_evictions=%llu recovered=%llu unrecovered=%llu\n",
            (unsigned long long)st_.vram_alloc_failures,
            (unsigned long long)st_.vram_alloc_retries,
            (unsigned long long)st_.vram_recovery_evictions,
            (unsigned long long)st_.vram_recovered_allocations,
            (unsigned long long)st_.vram_unrecovered_allocations);
}

// =============================================================================================
// ExpertPager implementation (callers hold the store mutex)
// =============================================================================================
inline bool ExpertPager::stage_to_ram(ExpertDescriptor& e) {
    auto t0 = Clock::now();
    if (e.state != Residency::SSD || e.pin_count) return false;
    S.begin(e, Residency::LOADING);
    const char* why = nullptr;
    void* p = nullptr;
    if (!S.ensure_ram_room(e.footprint, &e) || !(p = S.ram_alloc(e.footprint))) why = "ram capacity/alloc";
    if (!why) {
        S.ram_used_ += e.footprint;
        bool ok = !take_fault(S.fail_backing_copy);
        for (int c = 0; ok && c < 3; c++) {
            const TensorSlice& s = e.slice[c];
            if (s.file_offset + s.bytes > S.size_) { ok = false; break; }  // range must be completely present in the file
            memcpy((uint8_t*)p + e.res_off[c], S.backing(e, c), s.bytes);
            e.pending_mask |= 1 << c;
        }
        if (ok && take_fault(S.corrupt_backing_copy)) ((uint8_t*)p)[e.res_off[0] + e.bytes[0] / 2] ^= 0xA5;
        if (ok && S.cfg_.verify) ok = S.resident_hash(e, (const uint8_t*)p) == S.source_hash(e);
        if (!ok) { why = "copy/validate failed"; S.ram_used_ -= e.footprint; }
    }
    if (why) {
        char note[96];
        snprintf(note, sizeof note, "%s (partial mask=%x)", why, e.pending_mask);
        if (p) free(p);
        S.abort_transition(e);
        S.log_migration(e, SSD_RAM, Residency::SSD, Residency::RAM, 0, us_since(t0), false, note);
        return false;
    }
    e.ram_ptr = p;  // publish
    e.ram_mask = 7;
    e.pending_mask = 0;
    S.set_state(e, Residency::RAM);
    S.log_migration(e, SSD_RAM, Residency::SSD, Residency::RAM, e.total_bytes, us_since(t0), true, "");
    return true;
}

inline bool ExpertPager::stage_to_vram(ExpertDescriptor& e) {
    auto t0 = Clock::now();
    if (!S.dev_ || e.state != Residency::RAM || !e.ram_ptr || e.pin_count) return false;
    S.begin(e, Residency::LOADING);
    const char* why = nullptr;
    void* dp = nullptr;
    if (!S.ensure_vram_room(e.footprint, &e)) {
        why = "vram capacity";
    } else {
        dp = S.dev_->alloc(e.footprint);
        const bool allocation_failed = (dp == nullptr);
        if (allocation_failed) S.st_.vram_alloc_failures++;

        // A physical allocation can still fail after the free-memory check.
        // Reclaim one unpinned LRU resident at a time, then retry, bounded by
        // the number of descriptors. The source RAM copy of e remains valid.
        size_t retries = 0;
        while (!dp && retries < S.descs_.size()) {
            ExpertDescriptor* victim = S.lru_victim(Residency::VRAM, &e);
            if (!victim) break;

            retries++;
            if (!S.pager_->evict_vram_to_ram(*victim) &&
                !S.pager_->drop_vram(*victim)) {
                break;
            }
            S.st_.vram_recovery_evictions++;

            // Preserve the configured physical free-memory reserve on retries too.
            const size_t free = S.dev_->free_memory();
            if (free < e.footprint ||
                free - e.footprint < S.cfg_.vram_reserve) {
                continue;
            }

            S.st_.vram_alloc_retries++;
            dp = S.dev_->alloc(e.footprint);
            if (!dp) S.st_.vram_alloc_failures++;
        }

        if (allocation_failed) {
            if (dp) S.st_.vram_recovered_allocations++;
            else S.st_.vram_unrecovered_allocations++;
        }

        if (!dp) why = "vram alloc";
    }
    if (!why) {
        S.vram_used_ += e.footprint;
        bool ok = true;
        for (int c = 0; ok && c < 3; c++) {
            ok = S.dev_->h2d((uint8_t*)dp + e.res_off[c], (const uint8_t*)e.ram_ptr + e.res_off[c], e.bytes[c]);
            if (ok) e.pending_mask |= 1 << c;
        }
        ok = ok && S.dev_->sync();  // synchronisation point: device copy is valid only after this
        if (ok && S.cfg_.verify) ok = S.device_hash(e, dp) == S.source_hash(e);
        if (!ok) { why = "copy/validate failed"; S.vram_used_ -= e.footprint; }
    }
    if (why) {
        char note[96];
        snprintf(note, sizeof note, "%s (partial mask=%x)", why, e.pending_mask);
        if (dp) S.dev_->release(dp, e.footprint);
        S.abort_transition(e);  // RAM copy stays valid
        S.log_migration(e, RAM_VRAM, Residency::RAM, Residency::VRAM, 0, us_since(t0), false, note);
        return false;
    }
    e.vram_ptr = dp;  // publish new residency ...
    e.vram_mask = 7;
    e.pending_mask = 0;
    S.set_state(e, Residency::VRAM);
    S.ram_free(e);  // ... then release the old copy
    S.log_migration(e, RAM_VRAM, Residency::RAM, Residency::VRAM, e.total_bytes, us_since(t0), true, "");
    return true;
}

inline bool ExpertPager::evict_vram_to_ram(ExpertDescriptor& e) {
    auto t0 = Clock::now();
    if (e.state != Residency::VRAM || !e.vram_ptr || e.pin_count) return false;
    S.begin(e, Residency::EVICTING);
    const char* why = nullptr;
    void* p = nullptr;
    if (!S.ensure_ram_room(e.footprint, &e) || !(p = S.ram_alloc(e.footprint))) why = "ram capacity/alloc";
    if (!why) {
        S.ram_used_ += e.footprint;
        bool ok = true;
        for (int c = 0; ok && c < 3; c++) {
            ok = S.dev_->d2h((uint8_t*)p + e.res_off[c], (const uint8_t*)e.vram_ptr + e.res_off[c], e.bytes[c]);
            if (ok) e.pending_mask |= 1 << c;
        }
        ok = ok && S.dev_->sync();
        if (ok && S.cfg_.verify) ok = S.resident_hash(e, (const uint8_t*)p) == S.source_hash(e);
        if (!ok) { why = "copy/validate failed"; S.ram_used_ -= e.footprint; }
    }
    if (why) {
        char note[96];
        snprintf(note, sizeof note, "%s (partial mask=%x)", why, e.pending_mask);
        if (p) free(p);
        S.abort_transition(e);  // VRAM copy stays valid
        S.log_migration(e, VRAM_RAM, Residency::VRAM, Residency::RAM, 0, us_since(t0), false, note);
        return false;
    }
    e.ram_ptr = p;
    e.ram_mask = 7;
    e.pending_mask = 0;
    S.set_state(e, Residency::RAM);
    S.vram_free(e);
    S.log_migration(e, VRAM_RAM, Residency::VRAM, Residency::RAM, e.total_bytes, us_since(t0), true, "");
    return true;
}

inline bool ExpertPager::evict_ram_to_ssd(ExpertDescriptor& e) {
    auto t0 = Clock::now();
    if (e.state != Residency::RAM || e.pin_count) return false;
    S.begin(e, Residency::EVICTING);
    S.ram_free(e);  // GGUF backing is immutable and always valid: nothing to write back
    S.set_state(e, Residency::SSD);
    S.log_migration(e, RAM_SSD, Residency::RAM, Residency::SSD, 0, us_since(t0), true, "released");
    return true;
}

inline bool ExpertPager::drop_vram(ExpertDescriptor& e) {
    auto t0 = Clock::now();
    if (e.state != Residency::VRAM || e.pin_count) return false;
    S.begin(e, Residency::EVICTING);
    S.vram_free(e);
    S.set_state(e, Residency::SSD);
    S.log_migration(e, VRAM_SSD, Residency::VRAM, Residency::SSD, 0, us_since(t0), true, "dropped (RAM could not take copy)");
    return true;
}

}  // namespace eahmmf
