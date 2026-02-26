import argparse
import os
import time

from openai import OpenAI


def run_create(client: OpenAI, model: str, prompt: str):
    started = time.perf_counter()
    response = client.responses.create(model=model, input=prompt)
    elapsed = time.perf_counter() - started
    text = (response.output_text or "").strip()
    return {"total_s": elapsed, "first_token_s": elapsed, "text": text}


def run_stream(client: OpenAI, model: str, prompt: str):
    started = time.perf_counter()
    first_token_s = None
    final_response = None
    text_chunks = []
    with client.responses.stream(model=model, input=prompt) as stream:
        for event in stream:
            event_type = str(getattr(event, "type", "") or "")
            if first_token_s is None and event_type.startswith("response.output_text"):
                first_token_s = time.perf_counter() - started
            if event_type == "response.output_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    text_chunks.append(delta)
        final_response = stream.get_final_response()
    elapsed = time.perf_counter() - started
    text = "".join(text_chunks).strip() or (final_response.output_text or "").strip()
    return {"total_s": elapsed, "first_token_s": first_token_s or elapsed, "text": text}


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI Responses latency (create vs stream).")
    parser.add_argument("--model", default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--prompt", default="Give one short sentence about hydration and recovery.")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)
    runs = max(1, int(args.runs))
    create_total = []
    stream_total = []
    stream_ttfb = []

    print(f"Model: {args.model}")
    print(f"Runs: {runs}")
    print("")

    for i in range(runs):
        c = run_create(client, args.model, args.prompt)
        s = run_stream(client, args.model, args.prompt)
        create_total.append(c["total_s"])
        stream_total.append(s["total_s"])
        stream_ttfb.append(s["first_token_s"])
        print(
            f"Run {i + 1}: create total={c['total_s']:.3f}s | "
            f"stream first-token={s['first_token_s']:.3f}s total={s['total_s']:.3f}s"
        )

    def avg(values):
        return sum(values) / max(len(values), 1)

    print("")
    print(f"Avg create total: {avg(create_total):.3f}s")
    print(f"Avg stream first-token: {avg(stream_ttfb):.3f}s")
    print(f"Avg stream total: {avg(stream_total):.3f}s")
    if avg(create_total) > 0:
        improvement = (1.0 - (avg(stream_ttfb) / avg(create_total))) * 100.0
        print(f"First-token improvement vs create total: {improvement:.1f}%")


if __name__ == "__main__":
    main()
