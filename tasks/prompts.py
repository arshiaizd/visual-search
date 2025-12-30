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
    (
        "Check the image: does it include a green circle in any patch? "
        "Reply strictly with 'yes' or 'no'."
    ),
    # (
    #     "Describe the image in detail. What are the objects or shapes in the image, and how many are they?"
    # ),
    # (
    #     "Check the image: does it include a green circle in any patch? "
    #     "Reply with 'yes' or 'no' followed by a comma and the location of the green circle: "
    #     "one of 'up', 'down', 'left', 'right', 'center'. "
    #     "If there is no green circle, reply strictly with 'no'."
    # ),
    # (
    #     "Check the image: does it include a green circle in any patch? "
    #     "If yes, reply with 'yes' followed by a comma and the closest visible coordinate label to the green circle "
    #     "(copy the coordinate exactly as written). "
    #     "If there is no green circle, reply strictly with 'no'."
    # ),
    # (
    #     "Check the image: does it include a green circle in any patch? "
    #     "If yes, reply with 'yes' followed by a comma and the (x, y) coordinate of the green circle's center in pixels, "
    #     "with x increasing to the right and y increasing downward (format: '(x,y)'). "
    #     "If there is no green circle, reply strictly with 'no'."
    # ),
]


}
