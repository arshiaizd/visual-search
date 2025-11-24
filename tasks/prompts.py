PROMPT_FAMILIES = {
    # ------------------------------------------------
    # Coordinates task: return (r,c) pairs
    # ------------------------------------------------
    "find_green_circle": [
        (
            "Find the patches containing a green circle in a 10x10 grid. "
            "Reply ONLY with (r,c) pairs, e.g., (1,2) (3,4)."
        ),
        (
            "Which grid cells include a green circle? "
            "Use only (r,c) tuples."
        ),
        (
            "Locate every green circle patch. "
            "Output only (r,c) pairs separated by spaces."
        ),
    ],

    # ------------------------------------------------
    # Exists task: yes/no answer about presence of a green circle
    # ------------------------------------------------
    "exists_green_circle": [
        # (
        #     "Does this image contain at least one green circle? "
        #     "Answer only 'yes' or 'no'."
        # ),
        # (
        #     "Is there any green circle anywhere in this 10x10 grid? "
        #     "Respond with a single word: 'yes' or 'no'."
        # ),
        (
            "Check the image: does it include a green circle in any patch? "
            "Reply strictly with 'yes' or 'no'."
        ),
        # (
        #     "Look carefully at all 10x10 patches in the image. "
        #     "If you clearly see at least one green circle, answer 'yes'. "
        #     "If you do not clearly see any green circle, answer 'no'. "
        #     "Reply with exactly one word: 'yes' or 'no'."
        # ),
        # (
        #     "Decide whether the image contains a green circle. "
        #     "Base your answer only on what is visible. "
        #     "If you are not sure you see a green circle, answer 'no'. "
        #     "Respond with a single word: 'yes' or 'no'."
        # ),
    ]

}
