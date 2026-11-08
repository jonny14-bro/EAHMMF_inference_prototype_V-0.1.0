// test_eahmmf.cpp - validation suite for the expert residency layer (requirements section 4.13).
// Needs tiny.gguf (python3 test_tiny.py) and baseline/logits.txt (captured from the pre-EAHMMF binary).
#include <type_traits>

#define QWEN3MOE_NO_MAIN
#include "../qwen3moe.cpp"

using namespace eahmmf;

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...)                                                      \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(c)) {                                                        \
            g_fail++;                                                      \
            fprintf(stderr, "  FAIL %s:%d: %s  ", __FILE__, __LINE__, #c); \
            fprintf(stderr, " " __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                         \
        }                                                                  \
    } while (0)
#define SECTION(name) fprintf(stderr, "[%s]\n", name)

static const std::vector<int> PROMPT = {5, 17, 200, 33, 99, 3, 150, 256, 42, 77, 8, 9, 10, 11, 12};

struct Result {
    std::vector<float> logits;  // logits after the prompt
    std::vector<int> gen;       // greedy continuation
    std::vector<int> rid;       // routing trace
    std::vector<float> rw;
};
static bool same(const Result& a, const Result& b) {
    return a.logits.size() == b.logits.size() && !memcmp(a.logits.data(), b.logits.data(), a.logits.size() * 4) && a.gen == b.gen && a.rid == b.rid &&
           a.rw.size() == b.rw.size() && !memcmp(a.rw.data(), b.rw.data(), a.rw.size() * 4);
}
// Prompt (batched prefill, batch 4) + 12 greedy tokens, with routing trace. Store must already be configured.
static Result infer(Engine& E) {
    E.trace_routing = true;
    E.trace_ids.clear();
    E.trace_w.clear();
    Result r;
    r.logits.resize(E.n_vocab);
    E.eval(PROMPT, 0, 4, r.logits.data());
    int pos = (int)PROMPT.size();
    std::vector<float> lg = r.logits;
    for (int i = 0; i < 12; i++) {
        int t = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        r.gen.push_back(t);
        E.forward(&t, 1, pos++, lg.data());
    }
    r.rid = E.trace_ids;
    r.rw = E.trace_w;
    E.trace_routing = false;
    return r;
}
static Result run(Engine& E, const Config& cfg, std::unique_ptr<DeviceBackend> dev = nullptr) {
    if (!E.init_experts(cfg, std::move(dev))) {
        CHECK(false, "test setup rejected expert-store reinitialization");
        return {};
    }
    return infer(E);
}
static bool bytes_match_source(Engine& E, int l, int x, Residency* served = nullptr, bool* copied = nullptr) {
    ExpertLease lease = E.store->acquire_lease(l, x);
    if (!lease) return false;
    const ExpertViews& v = lease.views();
    const ExpertDescriptor& d = E.store->descriptor(l, x);
    const WeightView* w[3] = {&v.gate, &v.up, &v.down};
    bool ok = true, cp = true;
    for (int c = 0; c < 3; c++) {
        const uint8_t* src = E.g.map + d.slice[c].file_offset;
        ok &= !memcmp(w[c]->data, src, d.bytes[c]);
        cp &= w[c]->data != src;
    }
    if (served) *served = v.served_from;
    if (copied) *copied = cp;
    return ok;
}

// Device whose memory holds an XOR-scrambled image: reading a device pointer from the CPU yields garbage,
// so correct logits prove the store bounced the data through d2h instead.
struct OpaqueDevice : SimDevice {
    const char* name() const override { return "opaque"; }
    bool h2d(void* d, const void* s, size_t n) override {
        if (!SimDevice::h2d(d, s, n)) return false;
        for (size_t i = 0; i < n; i++) ((uint8_t*)d)[i] ^= 0xFF;
        return true;
    }
    bool d2h(void* d, const void* s, size_t n) override {
        if (!SimDevice::d2h(d, s, n)) return false;
        for (size_t i = 0; i < n; i++) ((uint8_t*)d)[i] ^= 0xFF;
        return true;
    }
    bool cpu_readable() const override { return false; }
};

// Simulated device with externally controllable physical free memory.
struct PhysicalPressureSimDevice : SimDevice {
    size_t free_bytes = 0;
    uint64_t alloc_attempts = 0;

    const char* name() const override { return "physical-pressure-sim"; }

    void* alloc(size_t n) override {
        alloc_attempts++;
        if (n > free_bytes) return nullptr;
        void* p = SimDevice::alloc(n);
        if (p) free_bytes -= n;
        return p;
    }

    void release(void* p, size_t n) override {
        if (!p) return;
        SimDevice::release(p, n);
        free_bytes += n;
    }

    // This becomes a DeviceBackend override when the interface is extended.
    size_t free_memory() const override { return free_bytes; }
};

int main(int argc, char** argv) {
    init_types();
    const char* model = argc > 1 ? argv[1] : "tiny.gguf";
    Engine E;
    E.load(model, 128, false);
    const size_t FP = E.store->descriptor(0, 0).footprint;

    SECTION("1. baseline: default (SSD tier) logits match the pre-EAHMMF binary");
    Result base;
    {
        std::vector<float> lg(E.n_vocab);
        E.eval(PROMPT, 0, 32, lg.data());
        FILE* f = fopen("baseline/logits.txt", "r");
        char buf[1024] = {0};
        CHECK(f && fgets(buf, sizeof buf, f), "baseline/logits.txt missing");
        if (f) fclose(f);
        // Baseline numbers were printed with %.6f by the pre-EAHMMF binary; allow 2e-6 so that other
        // compiler flags (FMA / vectorisation order) do not cause false alarms. Bitwise equality is
        // enforced between tiers *within* one binary below.
        const char* p = buf;
        int n = 0;
        double maxd = 0;
        for (int i = 0; i < 16; i++) {
            char* end;
            double want = strtod(p, &end);
            if (end == p) break;
            p = end;
            maxd = std::max(maxd, std::fabs(want - (double)lg[i]));
            n++;
        }
        CHECK(n == 16 && maxd < 2e-6, "baseline logits differ by %.2e (n=%d)", maxd, n);
        base = run(E, Config());
        CHECK(!base.rid.empty() && base.rid.size() == base.rw.size());
        CHECK(base.rid.size() == (size_t)E.n_layer * (PROMPT.size() + 12) * E.n_used, "trace has %zu entries", base.rid.size());
    }

    SECTION("2. descriptors: stable identity, slices match GGUF, sizes");
    {
        E.init_experts(Config());
        bool all = true;
        for (int l = 0; l < E.n_layer; l++)
            for (int x = 0; x < E.n_expert; x++) {
                const ExpertDescriptor& d = E.store->descriptor(l, x);
                const Tensor* ts[3] = {E.layers[l].gate, E.layers[l].up, E.layers[l].down};
                all &= d.layer == l && d.expert_id == x && d.state == Residency::SSD && d.pin_count == 0;
                size_t sum = 0;
                for (int c = 0; c < 3; c++) {
                    size_t rows = (size_t)ts[c]->ne[1];
                    all &= d.slice[c].file_offset == ts[c]->file_off + (uint64_t)x * rows * ts[c]->row_bytes;
                    all &= d.bytes[c] == rows * ts[c]->row_bytes && d.slice[c].type == ts[c]->type;
                    all &= E.g.map + d.slice[c].file_offset == ts[c]->data + (size_t)x * rows * ts[c]->row_bytes;
                    all &= d.res_off[c] % 64 == 0;
                    sum += d.bytes[c];
                }
                all &= sum == d.total_bytes && d.footprint >= d.total_bytes && d.footprint % 4096 == 0;
            }
        CHECK(all, "descriptor metadata mismatch");
        CHECK(E.store->count_in(Residency::SSD) == (size_t)E.n_layer * E.n_expert);
        ExpertViews v = E.store->acquire(1, 5);  // SSD-served view is zero-copy into the mmap
        CHECK(v.ok && v.served_from == Residency::SSD && v.gate.data == E.g.map + E.store->descriptor(1, 5).slice[0].file_offset);
        E.store->release(1, 5);
    }

    SECTION("3. tier matrix: logits, generation and routing identical (bitwise) to baseline");
    {
        struct Case { const char* name; Config cfg; };
        std::vector<Case> cases;
        auto mk = [&](Residency t, size_t ram, size_t vram, bool verify) { Config c; c.exec_tier = t; c.ram_budget = ram; c.vram_budget = vram; c.verify = verify; c.record_events = true; return c; };
        cases.push_back({"ram unlimited", mk(Residency::RAM, SIZE_MAX, SIZE_MAX, true)});
        cases.push_back({"vram unlimited", mk(Residency::VRAM, SIZE_MAX, SIZE_MAX, true)});
        cases.push_back({"ram, 3-expert budget", mk(Residency::RAM, 3 * FP, SIZE_MAX, true)});
        cases.push_back({"vram 2-expert, ram unlimited", mk(Residency::VRAM, SIZE_MAX, 2 * FP, true)});
        cases.push_back({"vram 2-expert, ram 1-expert", mk(Residency::VRAM, 1 * FP, 2 * FP, true)});
        cases.push_back({"vram 1-expert, ram 1-expert", mk(Residency::VRAM, 1 * FP, 1 * FP, true)});
        cases.push_back({"vram budget < 1 expert (all fallback)", mk(Residency::VRAM, SIZE_MAX, FP / 2, false)});
        for (auto& c : cases) {
            Result r = run(E, c.cfg);
            CHECK(same(r, base), "case '%s' differs from baseline", c.name);
            std::string inv = E.store->check_invariants();
            CHECK(inv.empty(), "%s: %s", c.name, inv.c_str());
            const Stats& st = E.store->stats();
            CHECK(st.acquires == st.releases && st.pinned_now == 0, "pin leak in '%s'", c.name);
            fprintf(stderr, "    %-40s ssd->ram=%llu ram->vram=%llu vram->ram=%llu ram->ssd=%llu fallbacks=%llu\n", c.name, (unsigned long long)st.mig_n[0],
                    (unsigned long long)st.mig_n[1], (unsigned long long)st.mig_n[2], (unsigned long long)st.mig_n[3], (unsigned long long)st.fallbacks);
        }
        {   // RAM can hold the incoming expert plus one evicted victim: every pager path is exercised
            Result r = run(E, mk(Residency::VRAM, 2 * FP, 2 * FP, true));
            const Stats& st = E.store->stats();
            CHECK(same(r, base));
            CHECK(st.mig_n[SSD_RAM] > 0 && st.mig_n[RAM_VRAM] > 0 && st.mig_n[VRAM_RAM] > 0 && st.mig_n[RAM_SSD] > 0, "constrained run should exercise every path");
            CHECK(E.store->ram_used() <= 2 * FP && E.store->vram_used() <= 2 * FP);
        }
        {   // RAM holds only the expert being promoted: VRAM victims cannot be written back and are dropped to SSD
            Result r = run(E, mk(Residency::VRAM, 1 * FP, 2 * FP, true));
            const Stats& st = E.store->stats();
            CHECK(same(r, base));
            CHECK(st.mig_n[VRAM_SSD] > 0 && st.mig_n[VRAM_RAM] == 0, "drop path used when RAM cannot take the copy");
        }
    }

    SECTION("4. residency walk SSD->RAM->VRAM->RAM->SSD with byte-exact data at each tier");
    {
        Config c;
        c.verify = true;
        E.init_experts(c);
        Residency sv;
        bool cp;
        CHECK(E.store->residency(0, 7) == Residency::SSD && bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::SSD && !cp);
        CHECK(E.store->promote(0, 7, Residency::RAM) && E.store->residency(0, 7) == Residency::RAM);
        CHECK(bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::RAM && cp);
        CHECK(E.store->promote(0, 7, Residency::VRAM) && E.store->residency(0, 7) == Residency::VRAM);
        CHECK(E.store->descriptor(0, 7).ram_ptr == nullptr && E.store->ram_used() == 0 && E.store->vram_used() == FP, "exclusive residency / accounting");
        CHECK(bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::VRAM && cp);
        CHECK(E.store->demote(0, 7, Residency::RAM) && E.store->residency(0, 7) == Residency::RAM && E.store->vram_used() == 0);
        CHECK(bytes_match_source(E, 0, 7, &sv) && sv == Residency::RAM);
        CHECK(E.store->demote(0, 7, Residency::SSD) && E.store->residency(0, 7) == Residency::SSD && E.store->ram_used() == 0);
        CHECK(bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::SSD && !cp);
        CHECK(E.store->promote(0, 7, Residency::VRAM), "direct SSD->VRAM goes through RAM");
        CHECK(E.store->check_invariants().empty());
    }

    SECTION("5. repeated acquire of a resident expert: hits, no migration, nested pins");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        E.init_experts(c);
        CHECK(E.store->acquire(1, 2).ok);
        E.store->release(1, 2);
        const Stats before = E.store->stats();
        ExpertViews a = E.store->acquire(1, 2), b = E.store->acquire(1, 2);
        CHECK(a.ok && b.ok && a.gate.data == b.gate.data && E.store->descriptor(1, 2).pin_count == 2);
        CHECK(!E.store->demote(1, 2, Residency::RAM), "pinned expert must not move");
        CHECK(!E.store->promote(1, 2, Residency::RAM) &&
              E.store->residency(1, 2) == Residency::VRAM &&
              E.store->descriptor(1, 2).pin_count == 2,
              "pinned expert must not be promoted");
        E.store->release(1, 2);
        CHECK(E.store->descriptor(1, 2).pin_count == 1 && !E.store->demote(1, 2, Residency::SSD));
        E.store->release(1, 2);
        const Stats& st = E.store->stats();
        CHECK(st.hit[2] == before.hit[2] + 2 && st.miss[2] == before.miss[2]);
        uint64_t mig = 0, mig0 = 0;
        for (int p = 0; p < 5; p++) { mig += st.mig_n[p]; mig0 += before.mig_n[p]; }
        CHECK(mig == mig0, "no migration on a resident hit");
        CHECK(st.pinned_now == 0 && !E.store->release(1, 2), "release of unpinned expert is rejected");
    }

    SECTION("6. two layers, different experts, different tiers");
    {
        E.init_experts(Config());
        CHECK(E.store->promote(0, 3, Residency::VRAM) && E.store->promote(1, 5, Residency::RAM) && E.store->promote(1, 3, Residency::RAM));
        CHECK(E.store->residency(0, 3) == Residency::VRAM && E.store->residency(1, 5) == Residency::RAM && E.store->residency(1, 3) == Residency::RAM);
        CHECK(E.store->residency(0, 5) == Residency::SSD && E.store->residency(1, 4) == Residency::SSD);
        CHECK(E.store->count_in(Residency::VRAM) == 1 && E.store->count_in(Residency::RAM) == 2);
        CHECK(bytes_match_source(E, 0, 3) && bytes_match_source(E, 1, 5) && bytes_match_source(E, 1, 3) && bytes_match_source(E, 0, 5));
        CHECK(E.store->descriptor(0, 3).slice[0].file_offset != E.store->descriptor(1, 3).slice[0].file_offset, "same expert id in different layers differs");
        CHECK(E.store->check_invariants().empty());
    }

    SECTION("7. VRAM capacity forces LRU eviction (VRAM->RAM->SSD)");
    {
        Config c;
        c.vram_budget = 2 * FP;
        c.ram_budget = 2 * FP;  // incoming expert's staging copy + one evicted victim
        c.verify = true;
        E.init_experts(c);
        CHECK(E.store->promote(0, 0, Residency::VRAM) && E.store->promote(0, 1, Residency::VRAM));
        CHECK(E.store->vram_used() == 2 * FP);
        CHECK(E.store->promote(0, 2, Residency::VRAM));  // evicts LRU = (0,0) to RAM
        CHECK(E.store->residency(0, 0) == Residency::RAM && E.store->residency(0, 1) == Residency::VRAM && E.store->residency(0, 2) == Residency::VRAM);
        CHECK(E.store->promote(0, 3, Residency::VRAM));  // evicts (0,1) to RAM, which pushes (0,0) RAM->SSD
        CHECK(E.store->residency(0, 0) == Residency::SSD && E.store->residency(0, 1) == Residency::RAM);
        CHECK(E.store->vram_used() <= 2 * FP && E.store->ram_used() <= 2 * FP && E.store->check_invariants().empty());
        for (int x = 0; x < 4; x++) CHECK(bytes_match_source(E, 0, x), "expert %d corrupt after eviction chain", x);
    }

    SECTION("8. in-use expert cannot be evicted; pool exhausted by pinned experts");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 2 * FP;
        E.init_experts(c);
        ExpertViews a = E.store->acquire(0, 0), b = E.store->acquire(0, 1);
        CHECK(a.ok && b.ok && a.served_from == Residency::VRAM && b.served_from == Residency::VRAM);
        ExpertViews cc = E.store->acquire(0, 2);  // non-strict: cannot evict pinned -> degraded service, still valid
        CHECK(cc.ok && cc.served_from == Residency::RAM, "falls back to its valid RAM copy");
        CHECK(E.store->residency(0, 0) == Residency::VRAM && E.store->residency(0, 1) == Residency::VRAM, "pinned experts stay put");
        CHECK(E.store->stats().refused_evictions > 0 && E.store->stats().fallbacks > 0);
        CHECK(!memcmp(cc.gate.data, E.g.map + E.store->descriptor(0, 2).slice[0].file_offset, E.store->descriptor(0, 2).bytes[0]));
        E.store->release(0, 2);
        E.store->set_strict(true);
        ExpertViews d = E.store->acquire(0, 3);
        CHECK(!d.ok, "strict mode refuses instead of degrading");
        CHECK(E.store->descriptor(0, 3).pin_count == 0, "failed acquire must not leave a pin");
        E.store->set_strict(false);
        E.store->release(0, 0);
        ExpertViews e = E.store->acquire(0, 2);  // (0,0) unpinned now -> LRU victim
        CHECK(e.ok && e.served_from == Residency::VRAM && E.store->residency(0, 0) == Residency::RAM && E.store->residency(0, 1) == Residency::VRAM);
        E.store->release(0, 2);
        E.store->release(0, 1);
        CHECK(E.store->check_invariants().empty() && E.store->stats().pinned_now == 0);
    }

    SECTION("9. failed staging never exposes invalid memory (fault injection)");
    {
        Config c;
        c.verify = true;
        E.init_experts(c);
        auto* sim = static_cast<SimDevice*>(E.dev.get());
        E.store->set_fault_injection(0, 1, 0);  // SSD->RAM faults
        CHECK(!E.store->promote(0, 0, Residency::RAM) && E.store->residency(0, 0) == Residency::SSD && E.store->ram_used() == 0);
        E.store->set_fault_injection(1, 0, 0);
        CHECK(!E.store->promote(0, 0, Residency::RAM) && E.store->residency(0, 0) == Residency::SSD && E.store->ram_used() == 0);
        E.store->set_fault_injection(0, 0, 1);
        CHECK(!E.store->promote(0, 0, Residency::RAM), "checksum must catch a corrupt staged copy");
        CHECK(E.store->residency(0, 0) == Residency::SSD && E.store->ram_used() == 0 && bytes_match_source(E, 0, 0));
        CHECK(E.store->promote(0, 0, Residency::RAM));
        sim->fail_alloc = 1;  // RAM->VRAM faults: the RAM copy must stay valid
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM && E.store->descriptor(0, 0).vram_ptr == nullptr);
        sim->fail_h2d = 1;
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM);
        sim->partial_h2d = 1;
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM && E.store->vram_used() == 0);
        sim->corrupt_h2d = 1;
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM && E.store->vram_used() == 0);
        CHECK(sim->live_bytes == 0, "failed device copies must free their allocation");
        CHECK(bytes_match_source(E, 0, 0) && E.store->check_invariants().empty());
        CHECK(E.store->promote(0, 0, Residency::VRAM) && sim->pending == 0, "sync point: no un-synchronised copies after publish");
        sim->fail_d2h = 1;  // VRAM->RAM fault: the VRAM copy must stay valid
        CHECK(!E.store->demote(0, 0, Residency::RAM) && E.store->residency(0, 0) == Residency::VRAM && bytes_match_source(E, 0, 0));
        CHECK(E.store->check_invariants().empty());

        Config c2;  // d2h failure during capacity-forced eviction: victim dropped to SSD, newcomer still lands in VRAM
        c2.vram_budget = 1 * FP;
        c2.verify = true;
        E.init_experts(c2);
        sim = static_cast<SimDevice*>(E.dev.get());
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        sim->fail_d2h = 1;
        CHECK(E.store->promote(0, 1, Residency::VRAM) && E.store->residency(0, 0) == Residency::SSD && E.store->residency(0, 1) == Residency::VRAM);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1) && E.store->check_invariants().empty());

        Config c3;  // intermittent faults throughout a full inference: result must still equal the baseline
        c3.exec_tier = Residency::VRAM;
        c3.vram_budget = 3 * FP;
        c3.ram_budget = 2 * FP;
        c3.verify = true;
        E.init_experts(c3);
        sim = static_cast<SimDevice*>(E.dev.get());
        sim->fail_alloc = 3; sim->fail_h2d = 4; sim->partial_h2d = 3; sim->corrupt_h2d = 2; sim->fail_d2h = 4;
        E.store->set_fault_injection(3, 3, 3);
        Result r = infer(E);
        CHECK(same(r, base), "inference with injected faults must match baseline");
        CHECK(E.store->stats().fallbacks > 0 && E.store->stats().mig_fail[0] + E.store->stats().mig_fail[1] > 0, "faults were actually hit");
        CHECK(E.store->check_invariants().empty() && sim->live_bytes == E.store->vram_used());
    }

    SECTION("10. physical VRAM pressure evicts before allocation");
    {
        Config c;
        c.vram_budget = 2 * FP;
        c.ram_budget = 2 * FP;
        c.verify = true;

        auto dev = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* sim = dev.get();
        sim->free_bytes = 2 * FP;
        CHECK(E.init_experts(c, std::move(dev)));

        CHECK(E.store->promote(0, 0, Residency::VRAM));
        CHECK(sim->free_bytes == FP && sim->alloc_attempts == 1);

        // Simulate another CUDA allocation consuming all remaining free memory.
        sim->free_bytes = 0;

        CHECK(E.store->promote(0, 1, Residency::VRAM),
              "pager should evict a VRAM resident before allocating under physical pressure");
        CHECK(sim->alloc_attempts == 2, "only the initial and successful incoming allocations attempted");
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::VRAM);
        CHECK(sim->free_bytes == 0 && sim->live_bytes == FP);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1));
        CHECK(E.store->check_invariants().empty());

        // A pinned resident cannot be evicted to satisfy physical pressure.
        auto dev2 = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* sim2 = dev2.get();
        sim2->free_bytes = 2 * FP;
        CHECK(E.init_experts(c, std::move(dev2)));
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        ExpertViews pinned = E.store->acquire(0, 0);
        CHECK(pinned.ok && pinned.served_from == Residency::VRAM);

        sim2->free_bytes = 0;
        CHECK(!E.store->promote(0, 1, Residency::VRAM),
              "must refuse when physical memory is low and every VRAM resident is pinned");
        CHECK(sim2->alloc_attempts == 1, "must refuse before another device allocation");
        CHECK(E.store->residency(0, 0) == Residency::VRAM &&
              E.store->residency(0, 1) == Residency::RAM);
        CHECK(E.store->descriptor(0, 0).pin_count == 1);
        CHECK(E.store->release(0, 0));
        CHECK(E.store->check_invariants().empty());

        // A configured physical reserve triggers eviction before allocation.
        Config reserve_cfg = c;
        reserve_cfg.vram_budget = 2 * FP;
        reserve_cfg.vram_reserve = FP / 2;
        auto reserve_dev = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* reserve_sim = reserve_dev.get();
        reserve_sim->free_bytes = 2 * FP;
        CHECK(E.init_experts(reserve_cfg, std::move(reserve_dev)));
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        CHECK(reserve_sim->free_bytes == FP);
        reserve_sim->free_bytes = FP + FP / 2 - 1;
        CHECK(E.store->promote(0, 1, Residency::VRAM),
              "reserve pressure should evict an unpinned resident before allocation");
        CHECK(reserve_sim->alloc_attempts == 2);
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::VRAM);
        CHECK(reserve_sim->free_bytes >= reserve_cfg.vram_reserve);
        CHECK(E.store->check_invariants().empty());

        // The reserve must not be consumed by evicting a pinned resident.
        auto reserve_dev2 = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* reserve_sim2 = reserve_dev2.get();
        reserve_sim2->free_bytes = 2 * FP;
        CHECK(E.init_experts(reserve_cfg, std::move(reserve_dev2)));
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        ExpertViews reserve_pinned = E.store->acquire(0, 0);
        CHECK(reserve_pinned.ok && reserve_pinned.served_from == Residency::VRAM);
        reserve_sim2->free_bytes = FP + FP / 2 - 1;
        CHECK(!E.store->promote(0, 1, Residency::VRAM),
              "reserve must be preserved when the only VRAM resident is pinned");
        CHECK(reserve_sim2->alloc_attempts == 1);
        CHECK(E.store->descriptor(0, 0).pin_count == 1);
        CHECK(E.store->release(0, 0));
        CHECK(E.store->check_invariants().empty());

        // Allocation fails once; reclaiming an LRU resident makes the retry succeed.
        CHECK(E.init_experts(c));
        auto* recovery_sim = static_cast<SimDevice*>(E.dev.get());
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        recovery_sim->fail_alloc = 1;
        CHECK(E.store->promote(0, 1, Residency::VRAM),
              "allocation should recover after evicting an unpinned VRAM resident");

        Stats recovered = E.store->stats();
        CHECK(recovered.vram_alloc_failures == 1);
        CHECK(recovered.vram_alloc_retries == 1);
        CHECK(recovered.vram_recovery_evictions == 1);
        CHECK(recovered.vram_recovered_allocations == 1);
        CHECK(recovered.vram_unrecovered_allocations == 0);
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::VRAM);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1));
        CHECK(E.store->check_invariants().empty());

        // Both allocation attempts fail; the incoming expert must remain in RAM.
        CHECK(E.init_experts(c));
        auto* failed_sim = static_cast<SimDevice*>(E.dev.get());
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        failed_sim->fail_alloc = 2;
        CHECK(!E.store->promote(0, 1, Residency::VRAM),
              "promotion must fail when the allocation retry also fails");

        Stats unrecovered = E.store->stats();
        CHECK(unrecovered.vram_alloc_failures == 2);
        CHECK(unrecovered.vram_alloc_retries == 1);
        CHECK(unrecovered.vram_recovery_evictions == 1);
        CHECK(unrecovered.vram_recovered_allocations == 0);
        CHECK(unrecovered.vram_unrecovered_allocations == 1);
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::RAM);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1));
        CHECK(failed_sim->live_bytes == 0);
        CHECK(E.store->check_invariants().empty());
    }

    SECTION("11. unavailable backing range -> ERROR state, never served");
    {
        std::vector<std::array<TensorRef, 3>> refs(E.n_layer);
        for (int l = 0; l < E.n_layer; l++) {
            const Tensor* ts[3] = {E.layers[l].gate, E.layers[l].up, E.layers[l].down};
            for (int c = 0; c < 3; c++) refs[l][c] = {ts[c]->file_off, ts[c]->type, ts[c]->ne[0], ts[c]->ne[1], ts[c]->ne[2], ts[c]->row_bytes};
        }
        refs[1][0].file_offset = E.g.size - 100;  // runs off the end of the file
        SimDevice sd;
        ExpertStore s(E.g.map, E.g.size, E.n_layer, E.n_expert, refs, Config(), &sd);
        CHECK(s.residency(1, 0) == Residency::ERROR && s.residency(0, 0) == Residency::SSD);
        CHECK(!s.acquire(1, 0).ok && !s.promote(1, 0, Residency::RAM) && s.check_invariants().empty());
        CHECK(s.acquire(0, 0).ok);
        s.release(0, 0);
    }

    SECTION("12. device memory the CPU cannot read (bounce path)");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 3 * FP;
        c.verify = true;
        Result r = run(E, c, std::unique_ptr<DeviceBackend>(new OpaqueDevice()));
        CHECK(same(r, base), "bounce-through-d2h result differs");
        CHECK(E.store->stats().mig_n[RAM_VRAM] > 0 && E.store->check_invariants().empty());
        bool none = true;
        for (int l = 0; l < E.n_layer; l++) for (int x = 0; x < E.n_expert; x++) none &= E.store->descriptor(l, x).bounce_ptr == nullptr;
        CHECK(none, "bounce buffers released on unpin");
    }

    SECTION("13. instrumentation: auditable events and counters");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 2 * FP;
        c.ram_budget = 2 * FP;
        c.record_events = true;
        c.log_path = "/tmp/eahmmf_test.log";
        Result r = run(E, c);
        CHECK(same(r, base));
        size_t acq = 0, rel = 0, mig = 0;
        uint64_t mbytes = 0;
        bool sane = true;
        for (auto& ev : E.store->events()) {
            if (ev.kind == Event::ACQUIRE) acq++;
            if (ev.kind == Event::RELEASE) rel++;
            if (ev.kind == Event::MIGRATE) {
                mig++;
                if (ev.ok) mbytes += ev.bytes;
                sane &= ev.from != ev.to && ev.layer >= 0 && ev.expert >= 0 && ev.ram_used <= 2 * FP && ev.vram_used <= 2 * FP;
            }
        }
        const Stats& st = E.store->stats();
        uint64_t sb = 0, sn = 0;
        for (int p = 0; p < 5; p++) { sb += st.mig_bytes[p]; sn += st.mig_n[p] + st.mig_fail[p]; }
        CHECK(acq == st.acquires && rel == st.releases && st.pins == st.unpins);
        CHECK(mig == sn && mbytes == sb && sane, "event stream agrees with counters");
        CHECK(st.hit[2] + st.miss[2] == st.acquires, "every acquire is a VRAM hit or miss");
        E.init_experts(Config());  // closes the log
        FILE* f = fopen("/tmp/eahmmf_test.log", "r");
        int lines = 0, macq = 0, mmig = 0;
        char buf[512];
        while (f && fgets(buf, sizeof buf, f)) { lines++; macq += strstr(buf, " ACQUIRE ") != nullptr; mmig += strstr(buf, " MIGRATE ") != nullptr; }
        if (f) fclose(f);
        CHECK(macq == (int)acq && mmig == (int)mig && lines > 0, "log file: %d acquire / %d migrate lines", macq, mmig);
    }


    SECTION("14. move-only ExpertLease: automatic pin ownership and release");
    {
        static_assert(!std::is_copy_constructible<ExpertLease>::value,
                      "ExpertLease must not be copy constructible");
        static_assert(!std::is_copy_assignable<ExpertLease>::value,
                      "ExpertLease must not be copy assignable");
        static_assert(std::is_move_constructible<ExpertLease>::value,
                      "ExpertLease must be move constructible");
        static_assert(std::is_move_assignable<ExpertLease>::value,
                      "ExpertLease must be move assignable");

        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 4 * FP;
        E.init_experts(c);

        // Scope exit must release exactly one pin.
        {
            ExpertLease lease = E.store->acquire_lease(0, 0);
            CHECK(static_cast<bool>(lease), "lease acquisition succeeds");
            CHECK(lease.views().ok, "lease exposes valid views");
            CHECK(E.store->descriptor(0, 0).pin_count == 1,
                  "lease owns one pin");
        }
        CHECK(E.store->descriptor(0, 0).pin_count == 0,
              "scope exit releases the pin");

        // Move construction transfers ownership; self-move is harmless.
        {
            ExpertLease source = E.store->acquire_lease(0, 1);
            CHECK(static_cast<bool>(source), "source lease acquired");

            ExpertLease destination(std::move(source));
            CHECK(!static_cast<bool>(source), "moved-from lease is empty");
            CHECK(static_cast<bool>(destination), "destination owns lease");
            CHECK(E.store->descriptor(0, 1).pin_count == 1,
                  "move construction preserves exactly one pin");

            destination = std::move(destination);
            CHECK(static_cast<bool>(destination),
                  "self-move assignment preserves ownership");
            CHECK(E.store->descriptor(0, 1).pin_count == 1,
                  "self-move does not alter pin count");

            destination.reset();
            destination.reset();
            CHECK(!static_cast<bool>(destination), "reset empties lease");
            CHECK(E.store->descriptor(0, 1).pin_count == 0,
                  "repeated reset releases only once");
        }

        // Move assignment releases the old destination pin before transfer.
        {
            ExpertLease destination = E.store->acquire_lease(0, 2);
            ExpertLease source = E.store->acquire_lease(0, 3);
            CHECK(destination && source, "both move-assignment leases acquired");

            destination = std::move(source);

            CHECK(!source, "move-assignment source becomes empty");
            CHECK(destination, "destination receives source ownership");
            CHECK(E.store->descriptor(0, 2).pin_count == 0,
                  "move assignment releases previous destination pin");
            CHECK(E.store->descriptor(0, 3).pin_count == 1,
                  "move assignment retains source pin exactly once");
        }
        CHECK(E.store->descriptor(0, 2).pin_count == 0 &&
              E.store->descriptor(0, 3).pin_count == 0,
              "scope exit releases moved destination pin");
        CHECK(E.store->stats().pinned_now == 0,
              "all lease pins released");

        // Failed strict acquisition must return an empty lease without a pin.
        Config strict_cfg;
        strict_cfg.exec_tier = Residency::VRAM;
        strict_cfg.vram_budget = 2 * FP;
        E.init_experts(strict_cfg);

        {
            ExpertLease first = E.store->acquire_lease(0, 0);
            ExpertLease second = E.store->acquire_lease(0, 1);
            CHECK(first && second, "two VRAM leases acquired");

            E.store->set_strict(true);
            ExpertLease refused = E.store->acquire_lease(0, 2);
            CHECK(!refused, "strict mode refuses unavailable lease");
            CHECK(E.store->descriptor(0, 2).pin_count == 0,
                  "failed lease acquisition leaves no pin");
        }

        CHECK(E.store->stats().pinned_now == 0,
              "strict-mode leases release cleanly");
        CHECK(E.store->check_invariants().empty(),
              "lease lifecycle preserves store invariants");
    }

    SECTION("15. store reinitialization is rejected while a lease is active");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 4 * FP;
        CHECK(E.init_experts(c), "initial store configuration succeeds");

        ExpertStore* const old_store = E.store.get();
        DeviceBackend* const old_dev = E.dev.get();

        {
            ExpertLease active = E.store->acquire_lease(0, 0);
            CHECK(static_cast<bool>(active), "active lease acquired");
            CHECK(E.store->stats().pinned_now == 1,
                  "store reports one pinned expert");

            CHECK(!E.init_experts(Config()),
                  "reinitialization is rejected while pinned");
            CHECK(E.store.get() == old_store,
                  "rejected reinitialization preserves original store");
            CHECK(E.dev.get() == old_dev,
                  "rejected reinitialization preserves original device");
            CHECK(E.store->descriptor(0, 0).pin_count == 1,
                  "rejected reinitialization preserves active pin");
        }

        CHECK(E.store->stats().pinned_now == 0,
              "lease release restores idle store");

        CHECK(E.init_experts(Config()),
              "reinitialization succeeds after lease release");
        CHECK(E.store != nullptr,
              "store remains available after successful reinitialization");
        CHECK(E.store->stats().pinned_now == 0,
              "replacement store starts without pins");
        CHECK(E.store->check_invariants().empty(),
              "replacement store invariants are valid");
    }

    fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
