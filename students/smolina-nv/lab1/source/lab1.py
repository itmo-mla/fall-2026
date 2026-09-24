import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)


FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)

C_INK, C_POS, C_NEG, C_ACC = '#141413', '#1D9E75', '#D85A30', '#C79A2E'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, name), dpi=130)
    plt.close(fig)


def header(text):
    print('\n' + '=' * 70)
    print(text)
    print('=' * 70)


header('ПОДГОТОВКА ДАННЫХ')

url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
       '00267/data_banknote_authentication.txt')
df = pd.read_csv(url, header=None,
                 names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
FEATURES = [c for c in df.columns if c != 'class']

print('Размер датасета:', df.shape)
print('Пропусков:', df.isna().sum().sum())
print('Баланс классов:')
print(df['class'].value_counts().to_string())

X = df[FEATURES].values
y = np.where(df['class'].values == 1, 1, -1)

# разбиение со стратификацией
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)


scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr_raw)
X_te = scaler.transform(X_te_raw)


X_tr = np.hstack([X_tr, -np.ones((X_tr.shape[0], 1))])
X_te = np.hstack([X_te, -np.ones((X_te.shape[0], 1))])

l, n = X_tr.shape
FEAT_NAMES = FEATURES + ['порог w0']
print('\ntrain: %s | test: %s | n = %d' % (X_tr.shape, X_te.shape, n))


def loss(M):
    return (1.0 - M) ** 2


def dloss(M):
    return -2.0 * (1.0 - M)


def grad_object(x, y_i, w):
    M = y_i * np.dot(w, x)
    return dloss(M) * y_i * x



def margins(X, y, w):
    return y * (X @ w)


def Q(X, y, w):
    return float(np.mean(loss(margins(X, y, w))))


def accuracy(X, y, w):
    pred = np.sign(X @ w)
    pred[pred == 0] = 1
    return float(np.mean(pred == y))


def optimal_step(x, c=1.0):
    norm2 = float(np.dot(x, x))
    if norm2 < 1e-12:
        raise ValueError('Нулевой объект: ||x||^2 = 0, шаг не определён')
    return c / (2.0 * norm2)



def margin_based_probs(X, y, w, eps=1e-3):
    M = margins(X, y, w)
    p = 1.0 / (np.abs(M) + eps)
    return p / p.sum()



def init_correlation(X, y):
    num = X.T @ y
    den = np.sum(X ** 2, axis=0)
    if np.any(den < 1e-12):
        raise ValueError('Константные признаки: %s' % np.where(den < 1e-12)[0].tolist())
    return num / den



def sgd_train(X, y, eta=0.01, gamma=0.5, tau=0.03,
              use_optimal_step=False, c=0.5,
              sampling='uniform',
              max_iter=20000, tol=1e-6, patience=500,
              seed=0, w0=None):
    l, n = X.shape
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1 / (2 * n), 1 / (2 * n), size=n) if w0 is None else w0.copy()
    v = np.zeros(n)

    lam = 1.0 / l
    Q_bar = Q(X, y, w)
    Q_hist = [Q_bar]
    probs, stable, stopped_at = None, 0, max_iter

    for t in range(max_iter):

        if sampling == 'margin' and t >= l and t % l == 0:
            probs = margin_based_probs(X, y, w)

        i = rng.integers(l) if probs is None else rng.choice(l, p=probs)

        M_i = y[i] * np.dot(w, X[i])
        L_i = loss(M_i)
        g = dloss(M_i) * y[i] * X[i]

        eta_t = optimal_step(X[i], c) if use_optimal_step else eta   # п.6

        w[:-1] *= (1.0 - eta_t * tau)
        v = gamma * v + eta_t * g
        w = w - v

        Q_prev = Q_bar
        Q_bar = (1 - lam) * Q_bar + lam * L_i
        Q_hist.append(Q_bar)

        if not np.isfinite(Q_bar):
            raise RuntimeError('Расходимость на итерации %d' % t)

        if abs(Q_bar - Q_prev) < tol:
            stable += 1
            if stable >= patience:
                stopped_at = t
                break
        else:
            stable = 0

    return w, np.array(Q_hist), stopped_at


def multistart(X, y, K=10, val_frac=0.25, seed=100, **kw):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_i, tr_i = idx[:n_val], idx[n_val:]

    best_w, best_score, scores = None, -np.inf, []
    for k in range(K):
        w, _, _ = sgd_train(X[tr_i], y[tr_i], seed=k, **kw)
        s = accuracy(X[val_i], y[val_i], w)
        scores.append(s)
        if s > best_score:
            best_score, best_w = s, w
    return best_w, np.array(scores)



header('ПУНКТ 1. ОТСТУП ОБЪЕКТА')


def analyze_margins(M, y_ref, title):
    neg = int((M < 0).sum())
    print('\n%s' % title)
    print('  средний отступ:        %+.4f' % M.mean())
    print('  min / max:             %+.4f / %+.4f' % (M.min(), M.max()))
    print('  доля ошибок (M < 0):   %.2f%%' % (100 * neg / len(M)))
    print('  доля на границе |M|<1: %.2f%%' % (100 * np.mean(np.abs(M) < 1)))
    print('  доля уверенных |M|>1:  %.2f%%' % (100 * np.mean(np.abs(M) > 1)))
    print('  зоны:')
    for nm, cnt in [('шумовые     M < -1', int((M < -1).sum())),
                    ('ошибочные   -1 <= M < 0', int(((M >= -1) & (M < 0)).sum())),
                    ('пограничные  0 <= M < 1', int(((M >= 0) & (M < 1)).sum())),
                    ('надёжные     M >= 1', int((M >= 1).sum()))]:
        print('    %-25s %4d (%.1f%%)' % (nm, cnt, 100 * cnt / len(M)))
    for cl in [-1, 1]:
        Mc = M[y_ref == cl]
        print('  класс %+d: средний отступ %+.4f, ошибок %.2f%%'
              % (cl, Mc.mean(), 100 * np.mean(Mc < 0)))


def plot_margins(M, title, fname):
    Ms = np.sort(M)
    x = np.arange(len(Ms))
    fig, ax = plt.subplots(figsize=(11, 5))
    for mask, color, lab in [(Ms < 0, C_NEG, 'ошибки (M < 0)'),
                             ((Ms >= 0) & (Ms < 1), C_ACC, 'пограничные (0 <= M < 1)'),
                             (Ms >= 1, C_POS, 'надёжные (M >= 1)')]:
        if mask.any():
            ax.fill_between(x[mask], Ms[mask], 0, color=color, alpha=0.45, label=lab)
    ax.plot(x, Ms, color=C_INK, lw=1.2)
    ax.axhline(0, color='black', lw=1)
    ax.axhline(1, color='gray', ls='--', lw=1)
    ax.set_xlabel('объекты, отсортированные по возрастанию отступа')
    ax.set_ylabel('отступ M')
    ax.set_title(title)
    ax.legend()
    save(fig, fname)


rng0 = np.random.default_rng(42)
w_rand = rng0.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
M_before = margins(X_tr, y_tr, w_rand)
analyze_margins(M_before, y_tr, 'До обучения (случайные веса), train:')
plot_margins(M_before, 'Отступы до обучения (случайные веса)', '1_margins_before.png')



header('ПУНКТ 8. ОБУЧЕНИЕ ЛИНЕЙНОГО КЛАССИФИКАТОРА')

ETA, GAMMA, TAU = 0.01, 0.5, 0.03
print('Общие параметры: eta = %.3f, gamma = %.1f, tau = %.3f' % (ETA, GAMMA, TAU))
print('Функция потерь квадратичная, SGD с инерцией, L2-регуляризация')


w_corr = init_correlation(X_tr, y_tr)
print('\n8.1 Инициализация весов через корреляцию')
print('  признак      |   w_j')
for j, nm in enumerate(FEAT_NAMES):
    print('  %-12s | %+8.4f' % (nm, w_corr[j]))
print('  accuracy на этих весах ДО обучения: %.4f' % accuracy(X_tr, y_tr, w_corr))

w_1, hist_1, stop_1 = sgd_train(X_tr, y_tr, eta=ETA, gamma=GAMMA, tau=TAU, w0=w_corr)
print('  после обучения: Q = %.4f, acc train = %.4f, итераций = %d'
      % (Q(X_tr, y_tr, w_1), accuracy(X_tr, y_tr, w_1), stop_1))


print('\n8.2 Мультистарт со случайной инициализацией (K = 10)')
w_2, ms_scores = multistart(X_tr, y_tr, K=10, eta=ETA, gamma=GAMMA, tau=TAU, max_iter=8000)
print('  accuracy на валидации: min %.4f | среднее %.4f (+-%.4f) | max %.4f'
      % (ms_scores.min(), ms_scores.mean(), ms_scores.std(), ms_scores.max()))
print('  лучшая модель: Q = %.4f, acc train = %.4f'
      % (Q(X_tr, y_tr, w_2), accuracy(X_tr, y_tr, w_2)))


print('\n8.3 Случайная инициализация + предъявление по модулю отступа')
w_3, hist_3, stop_3 = sgd_train(X_tr, y_tr, eta=ETA, gamma=GAMMA, tau=TAU,
                                sampling='margin')
print('  Q = %.4f, acc train = %.4f, итераций = %d'
      % (Q(X_tr, y_tr, w_3), accuracy(X_tr, y_tr, w_3), stop_3))


print('\nОбучение со скорейшим градиентным спуском (п.6, c = 0.5)')
w_4, hist_4, stop_4 = sgd_train(X_tr, y_tr, gamma=GAMMA, tau=TAU,
                                use_optimal_step=True, c=0.5)
print('  Q = %.4f, acc train = %.4f' % (Q(X_tr, y_tr, w_4), accuracy(X_tr, y_tr, w_4)))

models = {
    '8.1 корреляция': w_1,
    '8.2 мультистарт': w_2,
    '8.3 по модулю отступа': w_3,
    'скорейший спуск': w_4,
}


fig, ax = plt.subplots(figsize=(11, 5))
for (nm, h), c in zip([('8.1 корреляция', hist_1), ('8.3 по модулю отступа', hist_3),
                       ('скорейший спуск', hist_4)], [C_POS, C_NEG, C_ACC]):
    ax.plot(h, color=c, lw=1.3, label=nm)
ax.set_xlabel('итерация')
ax.set_ylabel(r'рекуррентная оценка $\bar{Q}$')
ax.set_title('Кривые обучения')
ax.legend()
save(fig, '2_learning_curves.png')



header('ПУНКТ 1. АНАЛИЗ ОТСТУПОВ ПОСЛЕ ОБУЧЕНИЯ')

M_after_tr = margins(X_tr, y_tr, w_1)
M_after_te = margins(X_te, y_te, w_1)
analyze_margins(M_after_tr, y_tr, 'После обучения, train:')
analyze_margins(M_after_te, y_te, 'После обучения, test:')
plot_margins(M_after_tr, 'Отступы после обучения (train)', '3_margins_after_train.png')
plot_margins(M_after_te, 'Отступы после обучения (test)', '4_margins_after_test.png')



header('ПУНКТ 9. ОЦЕНКА КАЧЕСТВА КЛАССИФИКАЦИИ')


def evaluate(X, y, w):
    scores = X @ w
    pred = np.sign(scores)
    pred[pred == 0] = 1
    return dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, pos_label=1, zero_division=0),
                recall=recall_score(y, pred, pos_label=1, zero_division=0),
                f1=f1_score(y, pred, pos_label=1, zero_division=0),
                roc_auc=roc_auc_score((y == 1).astype(int), scores))


print('%-24s | %-26s | %s' % ('', 'TRAIN', 'TEST'))
print('%-24s | %-26s | %s' % ('модель', 'acc     f1      auc', 'acc     f1      auc'))
print('-' * 82)
for nm, w in models.items():
    a, b = evaluate(X_tr, y_tr, w), evaluate(X_te, y_te, w)
    print('%-24s | %.4f  %.4f  %.4f    | %.4f  %.4f  %.4f'
          % (nm, a['accuracy'], a['f1'], a['roc_auc'],
             b['accuracy'], b['f1'], b['roc_auc']))

best_name = max(models, key=lambda k: accuracy(X_te, y_te, models[k]))
w_best = models[best_name]
print('\nЛучшая реализация: %s' % best_name)

m = evaluate(X_te, y_te, w_best)
print('Метрики на тестовой выборке:')
for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    print('  %-10s %.4f' % (k, m[k]))

pred_te = np.sign(X_te @ w_best)
pred_te[pred_te == 0] = 1
cm = confusion_matrix(y_te, pred_te, labels=[1, -1])
print('\nМатрица ошибок:')
print('              pred +1   pred -1')
print('  true +1     %7d   %7d' % (cm[0, 0], cm[0, 1]))
print('  true -1     %7d   %7d' % (cm[1, 0], cm[1, 1]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].imshow(cm, cmap='Greens')
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['pred +1', 'pred -1'])
ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(['true +1', 'true -1'])
for a_ in range(2):
    for b_ in range(2):
        ax[0].text(b_, a_, cm[a_, b_], ha='center', va='center', fontsize=15,
                   color='white' if cm[a_, b_] > cm.max() / 2 else 'black')
ax[0].set_title('Матрица ошибок (test)')

for nm, w in models.items():
    fpr, tpr, _ = roc_curve((y_te == 1).astype(int), X_te @ w)
    ax[1].plot(fpr, tpr, lw=1.3,
               label='%s (%.4f)' % (nm, roc_auc_score((y_te == 1).astype(int), X_te @ w)))
ax[1].plot([0, 1], [0, 1], '--', color='gray', lw=1)
ax[1].set_xlabel('FPR'); ax[1].set_ylabel('TPR')
ax[1].set_title('ROC-кривые (test)')
ax[1].legend(fontsize=8, loc='lower right')
save(fig, '5_metrics.png')



header('ПУНКТ 10. СРАВНЕНИЕ С ЭТАЛОННОЙ РЕАЛИЗАЦИЕЙ')

ref = SGDClassifier(loss='squared_error', penalty='l2', alpha=TAU,
                    learning_rate='optimal', max_iter=1000, tol=1e-4,
                    fit_intercept=False, random_state=42).fit(X_tr, y_tr)
w_ref = ref.coef_[0]

e_my, e_ref = evaluate(X_te, y_te, w_best), evaluate(X_te, y_te, w_ref)
print('%-24s | accuracy | precision | recall |   f1   | roc_auc' % 'реализация')
print('-' * 78)
for nm, e in [('моя (%s)' % best_name, e_my), ('эталон SGDClassifier', e_ref)]:
    print('%-24s | %8.4f | %9.4f | %6.4f | %6.4f | %7.4f'
          % (nm, e['accuracy'], e['precision'], e['recall'], e['f1'], e['roc_auc']))

print('\nРазница (моя - эталонная):')
for k in ['accuracy', 'precision', 'recall', 'f1']:
    print('  %-10s %+.4f' % (k, e_my[k] - e_ref[k]))

cos = float(np.dot(w_best, w_ref) / (np.linalg.norm(w_best) * np.linalg.norm(w_ref)))
print('\nКосинус угла между векторами весов: %.4f' % cos)
print('(классификатор инвариантен к масштабу w, поэтому сравнивается направление)')

print('\nВеса, нормированные на ||w||:')
print('  признак      |     моя |  эталон')
wn = w_best / np.linalg.norm(w_best)
rn = w_ref / np.linalg.norm(w_ref)
for j, nm in enumerate(FEAT_NAMES):
    print('  %-12s | %+7.4f | %+7.4f' % (nm, wn[j], rn[j]))

fig, ax = plt.subplots(figsize=(9, 4.5))
xp = np.arange(n)
ax.bar(xp - 0.2, wn, 0.4, label='моя реализация', color=C_POS)
ax.bar(xp + 0.2, rn, 0.4, label='SGDClassifier', color=C_NEG)
ax.set_xticks(xp); ax.set_xticklabels(FEAT_NAMES, rotation=15, fontsize=9)
ax.axhline(0, color='black', lw=.8)
ax.set_ylabel(r'$w_j / \|w\|$')
ax.set_title('Направление вектора весов (косинус %.4f)' % cos)
ax.legend()
save(fig, '6_reference.png')


header('СВОДНАЯ ТАБЛИЦА')

rows = []
for nm, w in list(models.items()) + [('эталон SGDClassifier', w_ref)]:
    e = evaluate(X_te, y_te, w)
    rows.append(dict(модель=nm, accuracy=round(e['accuracy'], 4),
                     precision=round(e['precision'], 4), recall=round(e['recall'], 4),
                     f1=round(e['f1'], 4), roc_auc=round(e['roc_auc'], 4)))

summary = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
print(summary.to_string(index=False))
summary.to_csv('summary.csv', index=False, encoding='utf-8')
print('\nСводка сохранена в summary.csv, графики — в папке %s/' % FIGDIR)
