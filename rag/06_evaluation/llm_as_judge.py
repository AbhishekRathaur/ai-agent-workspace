import json
import litellm


def calculate_faithfulness(context: str, answer: str) -> float:
    """
    Compute a faithfulness score for an answer given a context.

    Faithfulness ≈ fraction of factual claims in the answer
    that are explicitly supported by the context.

    Score in [0.0, 1.0]:
        1.0  => all claims are supported
        0.0  => no claims are supported or no claims extracted
    """
    # Pass 1: Extract individual claims from the generated answer
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Break down the text into an array of individual, distinct factual claims.\n"
                "Output ONLY a valid JSON array of strings: ['claim 1', 'claim 2']."
            ),
        },
        {"role": "user", "content": f"Text: {answer}"},
    ]

    extraction_res = litellm.completion(
        model="ollama/llama3.2",
        messages=extraction_prompt,
        api_base="http://localhost:11434",
        temperature=0.0,
    )

    raw_claims = extraction_res.choices[0].message.content.strip()

    try:
        claims = json.loads(raw_claims)
        if not isinstance(claims, list):
            # If model didn't follow JSON-array format, treat as no claims
            return 0.0
    except json.JSONDecodeError:
        # If JSON parsing fails, return 0.0 to avoid crashing
        return 0.0

    if not claims:
        return 0.0

    verified_count = 0

    # Pass 2: Verify each individual claim against the source context
    for claim in claims:
        verification_prompt = [
            {
                "role": "system",
                "content": (
                    f"Context: {context}\n\n"
                    "Analyze if the user's statement is explicitly supported by the context.\n"
                    "Respond with exactly one character: 'Y' if supported, 'N' if not supported. "
                    "Do not explain."
                ),
            },
            {"role": "user", "content": f"Statement: {claim}"},
        ]

        verification_res = litellm.completion(
            model="ollama/llama3.2",
            messages=verification_prompt,
            api_base="http://localhost:11434",
            temperature=0.0,
        )

        verdict = verification_res.choices[0].message.content.strip().upper()

        # Be strict: only count a clean 'Y' as supported
        if verdict == "Y":
            verified_count += 1

    # Faithfulness = verified claims / total claims
    return verified_count / len(claims)


if __name__ == "__main__":
    mock_context = (
        "MeshQuery routes requests across cluster nodes using an ultra-low latency "
        "RSocket protocol layer over TCP."
    )

    # Introduce a simulated hallucination about encryption standards
    mock_hallucinated_answer = (
        "MeshQuery handles communication using the RSocket protocol. "
        "It also securely encrypts all data packets using enterprise-grade AES-256 "
        "bits standards."
    )

    score = calculate_faithfulness(mock_context, mock_hallucinated_answer)

    print("\n===========================================")
    print(f"AUTOMATED FAITHFULNESS METRIC SCORE: {score:.2f} / 1.00")
    print("===========================================")

    if score < 1.0:
        print("🚨 ALERT: Hallucinated claims detected! Threat intercepted.")