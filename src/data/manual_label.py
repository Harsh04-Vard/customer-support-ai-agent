import pandas as pd

INPUT_FILE = "data/golden/intent_sample_200.csv"

INTENTS = {
    "1": "Battery",
    "2": "Software / OS Update",
    "3": "Keyboard / Input",
    "4": "App Problem",
    "5": "Device Performance",
    "6": "Connectivity",
    "7": "Calls / Audio",
    "8": "Messaging / Photos",
    "9": "Apple Services / Media",
    "10": "Account / Security",
    "11": "Purchase / Billing / Store",
    "12": "Hardware / Accessories",
    "13": "Other / Unclear",
}

# Load the dataset
df = pd.read_csv(INPUT_FILE)

# Make sure intent can store text values
df["intent"] = df["intent"].astype("object")

# Show 25 examples at a time
BATCH_SIZE = 25

while True:

    # Find all unlabeled rows
    unlabeled = df[
        df["intent"].isna()
        | (df["intent"].astype(str).str.strip() == "")
    ]

    # Stop when everything is labeled
    if len(unlabeled) == 0:
        print("\n🎉 All 200 examples have been labelled!")
        break

    # Take the next 25
    batch = unlabeled.head(BATCH_SIZE)

    print("\n" + "=" * 80)
    print(
        f"Examples {batch.index[0] + 1} - "
        f"{batch.index[-1] + 1}"
    )
    print("=" * 80)

    # Display examples
    for number, (i, row) in enumerate(batch.iterrows(), start=1):

        print(f"\nExample {number} "
              f"(Dataset row {i + 1}):")

        print(row["customer_text"])

    # Display intent options
    print("\n" + "-" * 80)
    print("INTENTS:")
    print("-" * 80)

    for key, value in INTENTS.items():
        print(f"{key}. {value}")

    # Ask for 25 labels
    print("\n" + "-" * 80)
    print(f"Enter {len(batch)} intent numbers separated by spaces.")
    print("Example:")
    print("1 2 3 13 7 1 8 4 10 13 ...")
    print("-" * 80)

    while True:

        choices = input(
            f"\nYour {len(batch)} labels: "
        ).strip().split()

        # Check number of labels
        if len(choices) != len(batch):
            print(
                f"❌ Please enter exactly "
                f"{len(batch)} numbers."
            )
            print(
                f"You entered {len(choices)}."
            )
            continue

        # Check valid intent numbers
        if not all(choice in INTENTS for choice in choices):
            print("❌ Use only numbers from 1 to 13.")
            continue

        # Everything is valid
        break

    # Save labels
    for i, choice in zip(batch.index, choices):
        df.at[i, "intent"] = INTENTS[choice]

    # Save after every batch
    df.to_csv(INPUT_FILE, index=False)

    print("\n✅ Batch saved!")

    # Calculate remaining examples
    remaining = len(unlabeled) - len(batch)

    print(
        f"Remaining examples: {remaining}"
    )

    print(
        f"Labeled so far: {len(df) - remaining}"
    )