import pandas as pd
import numpy as np
import logging
import sys
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split


#لاگر

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler('lightfm_fast_scenarios.log')
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


# 1) بارگذاری داده

df = pd.read_csv('order-Product_prior.csv')

# نسخه سریع
df = df.head(50000)  # فقط ۵۰هزار رکورد

# نسخه کامل و کند (اجرا طولانی ولی با دقت تخمین بالا) - در صورت نیاز فعال کنید
# df = pd.read_csv('order-Product_prior.csv')

df.columns = ['user_id', 'product_id', 'add_to_cart_order', 'reordered']
df['rating'] = df['reordered'].fillna(0).clip(0, 1).astype(float)

# نگاشت ID ها به ایندکس عددی
user_ids = df['user_id'].unique()
item_ids = df['product_id'].unique()
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}

df['u_idx'] = df['user_id'].map(user_map)
df['i_idx'] = df['product_id'].map(item_map)

# ساخت ماتریس تعاملات
interactions = coo_matrix(
    (df['rating'], (df['u_idx'], df['i_idx'])),
    shape=(len(user_ids), len(item_ids))
)
logger.info(f"ابعاد ماتریس تعاملات: {interactions.shape}")


# 2) تقسیم آموزش و تست

train, test = random_train_test_split(
    interactions, test_percentage=0.2, random_state=np.random.RandomState(42)
)


# 3) آموزش مدل

# نسخه سریع
model = LightFM(no_components=8, learning_rate=0.05, loss='warp')
model.fit(train, epochs=5, num_threads=4)

# نسخه کامل و کند (در صورت نیاز فعال کنید)
# model = LightFM(no_components=32, learning_rate=0.05, loss='warp')
# model.fit(train, epochs=20, num_threads=4)

logger.info("مدل آموزش داده شد.")


# 4) تابع توصیه برای یک کاربر

def recommend_for_user(model, train_mat, user_idx, n=5):
    n_users, n_items = train_mat.shape
    scores = model.predict(
        np.repeat(user_idx, n_items),
        np.arange(n_items),
        num_threads=4
    )
    known_items = set(train_mat.tocsr()[user_idx].indices)
    candidates = [i for i in range(n_items) if i not in known_items]
    top_idx = np.argsort(-scores[candidates])[:n]
    return [candidates[i] for i in top_idx], scores


# 5) سناریوهای سه‌مرحله‌ای

scenario_results = []
target_user = df['u_idx'].iloc[0]
user_all_items = set(train.tocsr()[target_user].indices) | set(test.tocsr()[target_user].indices)

for remove_n in [1, 2, 3]:
    if len(user_all_items) <= remove_n:
        logger.warning(f"کاربر آیتم کافی برای حذف {remove_n} ندارد.")
        continue

    removed_items = list(user_all_items)[:remove_n]
    train_mod = train.tolil(copy=True)
    for it in removed_items:
        train_mod[target_user, it] = 0.0

    # نسخه سریع
    model_mod = LightFM(no_components=8, learning_rate=0.05, loss='warp')
    model_mod.fit(train_mod, epochs=5, num_threads=4)

    # نسخه کامل و کند
    # model_mod = LightFM(no_components=32, learning_rate=0.05, loss='warp')
    # model_mod.fit(train_mod, epochs=20, num_threads=4)

    top_items, scores = recommend_for_user(model_mod, train_mod, target_user, n=5)
    correct_count = sum(1 for it in removed_items if it in top_items)
    accuracy = correct_count / remove_n
    logger.info(f"سناریو حذف {remove_n} آیتم - دقت: {accuracy:.2f}")

    for it in top_items:
        scenario_results.append({
            'scenario': f'حذف {remove_n} آیتم',
            'user_id': user_ids[target_user],
            'product_id': item_ids[it],
            'predicted_score': scores[it],
            'is_target_item': item_ids[it] in [item_ids[x] for x in removed_items]
        })

pd.DataFrame(scenario_results).to_excel('scenario_results.xlsx', index=False)


# 6) پیشنهاد برای همه کاربران

def get_top_n_all_users(model, train_mat, user_map, item_map, n=5):
    n_users, n_items = train_mat.shape
    all_results = []
    item_rev_map = {v: k for k, v in item_map.items()}
    for uid, uidx in user_map.items():
        scores = model.predict(np.repeat(uidx, n_items),
                               np.arange(n_items),
                               num_threads=4)
        known_items = set(train_mat.tocsr()[uidx].indices)
        candidates = [i for i in range(n_items) if i not in known_items]
        top_idx = np.argsort(-scores[candidates])[:n]
        for i in top_idx:
            all_results.append({
                'user_id': uid,
                'product_id': item_rev_map[candidates[i]],
                'predicted_score': scores[candidates[i]]
            })
    return all_results

full_recs = get_top_n_all_users(model, train, user_map, item_map, n=5)
pd.DataFrame(full_recs).to_excel('all_users_results.xlsx', index=False)
