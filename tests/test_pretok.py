import regex, random, subprocess
pat = regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
random.seed(1)
alpha = list("abcXYZ019 \t\n\r'\"!?.,;:(){}[]<>/\\_-+=*&^%$#@~`|") + ["'s","'RE","'ll","'d","  ","\n\n","\r\n"," \n ","é","ß","中","文","日本","٣","½","²","—","“","”","😀","🎉","\u00a0","\u3000","\u2003","Ω","ℵ","Ⅷ","०"]
bad = 0
for it in range(3000):
    s = "".join(random.choice(alpha) for _ in range(random.randint(1,30)))
    exp = pat.findall(s)
    out = subprocess.run(["./qwen3moe","-m","x","--pretok",s],capture_output=True).stdout.decode("utf8","replace")
    got = out[:-1].split("\x1f")[:-1] if out.strip("\n") else []
    if got != exp:
        bad += 1
        if bad < 6: print("MISMATCH", repr(s), "\n exp", exp, "\n got", got)
print("mismatches:", bad, "/ 3000")
