import os
import pandas as pd
from sklearn.model_selection import train_test_split

# configuration
INPUT_CSV   = 'data_matching/matched_image_tabular.csv'
IMAGE_ROOT  = 'preprocess_img/img_preprocess_output'
TRAIN_CSV   = 'data_matching/train_split.csv'
TEST_CSV    = 'data_matching/test_split.csv'
TEST_SIZE   = 0.2
RANDOM_SEED = 42

df = pd.read_csv(INPUT_CSV)
print("Columns in metadata:", df.columns.tolist())

path_col = 'npy_path'  
if path_col not in df.columns:
    raise KeyError(f"Expected '{path_col}' in {INPUT_CSV} but found {df.columns.tolist()}")

df['full_path'] = df[path_col].apply(lambda p: os.path.join(IMAGE_ROOT, p))
df = df[df['full_path'].apply(os.path.exists)]
print(f"Filtered to {len(df)} existing files out of {len(df)} total")

# Splitting
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

# save under data_matching/
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"✅ Train/test split complete.")
print(f"   Train: {len(train_df)} rows → {TRAIN_CSV}")
print(f"   Test:  {len(test_df)} rows → {TEST_CSV}")
