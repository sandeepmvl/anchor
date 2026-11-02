# Sample — run it (pip install tiktoken) to print YOUR counts for the ticket's segments; the numbers are not predicted here
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # the GPT-3.5/GPT-4-era vocabulary
ticket = "Order 48213: charged $1,249.99 twice. 请尽快退款。"
segments = ["Order", " 48213", " $1,249.99", " twice.", " 请尽快退款。"]

for seg in segments:
    ids = enc.encode(seg)
    # decode each id back so you can SEE where the cuts fall (partial-byte CJK pieces show as the replacement character)
    pieces = [enc.decode([i]) for i in ids]
    print(f"{seg!r}: {len(seg)} chars -> {len(ids)} tokens {pieces}")

print(f"whole ticket: {len(ticket)} chars -> {len(enc.encode(ticket))} tokens")
