import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
    
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','class','level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test  = pd.read_csv(test_url,  names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'],  inplace=True)

print(f"Train: {df_train.shape[0]:,} records | Test: {df_test.shape[0]:,} records")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers

cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    # DoS
    'neptune':'DoS','back':'DoS','land':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','mailbomb':'DoS','apache2':'DoS','processtable':'DoS',
    'udpstorm':'DoS','worm':'DoS',
    # Probe
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'warezclient':'R2L','guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L',
    'phf':'R2L','multihop':'R2L','warezmaster':'R2L','spy':'R2L','xlock':'R2L',
    'xsnoop':'R2L','snmpguess':'R2L','snmpgetattack':'R2L','httptunnel':'R2L',
    'sendmail':'R2L','named':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R','perl':'U2R',
    'sqlattack':'U2R','xterm':'U2R','ps':'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')
df_full.drop(columns=['num_outbound_cmds','class'], inplace=True)

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Split back into train and test
train_len = len(df_train)
df_tr = df_full.iloc[:train_len].copy()
df_te = df_full.iloc[train_len:].copy()

X_train = df_tr.drop(columns=['category'])
y_train = df_tr['category']
X_test  = df_te.drop(columns=['category'])
y_test  = df_te['category']

print("\nTraining class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================

label_order = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
label2int = {l: i for i, l in enumerate(label_order)}
int2label = {i: l for l, i in label2int.items()}

y_train_int = y_train.map(label2int)
y_test_int  = y_test.map(label2int)

print("\nApplying SMOTE to oversample minority classes...")

smote_strategy = {
    label2int['R2L']: 5000,
    label2int['U2R']: 2000,
}

smote = SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train_int)

print("Post-SMOTE class distribution:")
unique, counts = np.unique(y_res, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {int2label[u]}: {c:,}")

#  ===========================
#   XGBOOST WITH CLASS WEIGHT
#  ===========================

print("\nTraining XGBoost classifier...")

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_res)

model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist',
)

model.fit(X_res, y_res, sample_weight=sample_weights)
print("Training complete.")

#  ===========================
# CROSS VALIDATION
#  ===========================


print("\nRunning cross-validation on training set...")

from imblearn.pipeline import Pipeline as ImbPipeline

cv_pipeline = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE, k_neighbors=5)),
    ('clf', XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
    ))
])

cv_scores = cross_val_score(
    cv_pipeline, X_train, y_train_int,
    cv=5, scoring='f1_macro', n_jobs=-1
)
print(f"CV Macro F1:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#  ===========================
# EVALUATION
#  ===========================

y_pred_int = model.predict(X_test)
y_pred = [int2label[i] for i in y_pred_int]
y_test_labels = [int2label[i] for i in y_test_int]

macro_f1 = f1_score(y_test_labels, y_pred, average='macro')
print(f"\nTest  Macro F1: {macro_f1:.4f}")
print(f"\nCV vs Test gap: {cv_scores.mean() - macro_f1:+.4f}")

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=label_order))

#  ===========================
# MATRIX
#  ===========================

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()