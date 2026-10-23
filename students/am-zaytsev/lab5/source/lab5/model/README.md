
**Размерности:**
*   $X \in \mathbb{R}^{\ell \times n}$ — матрица признаков
*   $W \in \mathbb{R}^{n \times 1}$ — вектор весов
*   $Y \in \mathbb{R}^{\ell \times 1}$ — вектор ответов

**Функционал потерь (Log-loss):**

$$
L = \sum_{i=1}^{\ell} \log(1 + e^{-M_i})
$$

**Отступ (Margin):**

$$
M_i = Y_i (X_i W^T) \equiv Y_i X_i^T W
$$

Метод Ньютона-Рафсона:

$$
w_{t+1} = w_t - h (H)^{-1} \cdot J = w_t - h \left( \frac{\partial^2 L}{\partial w_j \partial w_k} \right)^{-1} \left(\frac{\partial L}{\partial w_j}\right)^T
$$

---

### Шаг 1. Вычисление градиента (первых производных)

$$
\frac{\partial L}{\partial w_j} = \sum_{i=1}^{\ell} \frac{1}{1 + e^{-M_i}} \cdot e^{-M_i} \cdot (-1) \cdot (Y_i X_{ij})
$$

Вводим сигмоиду $\sigma_i = \frac{1}{1 + e^{-M_i}}$, тогда $\frac{e^{-M_i}}{1 + e^{-M_i}} = 1 - \sigma_i$.

Получаем **(пункт ①):**

$$
\frac{\partial L}{\partial w_j} = -\sum_{i=1}^{\ell} (1 - \sigma_i) Y_i X_{ij}
$$

---

### Шаг 2. Вычисление Гессиана (вторых производных)

Берем производную от градиента:

$$
\frac{\partial^2 L}{\partial w_j \partial w_k} = -\sum_{i=1}^{\ell} Y_i X_{ij} \cdot (-1) \cdot (\sigma_i (1 - \sigma_i)) \cdot Y_i X_{ik}
$$

Так как $Y_i \in \{-1, 1\}$, то $Y_i^2 = 1$. Упрощаем **(пункт ②):**

$$
\frac{\partial^2 L}{\partial w_j \partial w_k} = \sum_{i=1}^{\ell} \sigma_i (1 - \sigma_i) X_{ij} X_{ik}
$$

---

### Шаг 3. Переход к матричным обозначениям

Введем диагональную матрицу весов:

$$
D = \text{diag}((1 - \sigma_i)\sigma_i)_{\ell \times \ell}
$$

Введем модифицированный вектор ответов:

$$
\tilde{Y} = \frac{Y_i}{\sigma_i}
$$

Тогда якобиан (пункт ①) и Гессиан (пункт ②) в матричном виде:
*   **Якобиан:** $J = \left(\nabla L \right)^T = (\frac{\partial L}{\partial w})^T = \left(-\tilde{Y}^T D X \right)^T= -X^T D \tilde{Y}$
*   **Гессиан:** $H = X^T D X$

Подставляем в формулу Ньютона:

$$
\left( \frac{\partial^2 L}{\partial w_j \partial w_k} \right)^{-1} \frac{\partial L}{\partial w_j} = (X^T D X)^{-1} (-X^T D \tilde{Y})
$$

---

### Шаг 4. Сведение к IRLS (взвешенному МНК)

Чтобы привести к виду $\tilde{X}^T \tilde{X}$, вводим $\sqrt{D}$:

$$
H = X^T \sqrt{D} \sqrt{D} X
$$

Вводим преобразованную матрицу признаков:

$$
\tilde{X} := \sqrt{D} X \quad (\text{размерность } \ell \times n)
$$

Модифицируем вектор ответов:

$$
\sqrt{D}\tilde{Y} = Y^* := \sqrt{D} \frac{Y_i}{\sigma_i} = \text{diag}\left(\sqrt{\sigma_i(1-\sigma_i)}\right) \frac{Y_i}{\sigma_i} = \sqrt{\frac{1-\sigma_i}{\sigma_i}} Y_i
$$

Собираем итоговое выражение:

$$
\Delta w = -(\tilde{X}^T \tilde{X})^{-1} \tilde{X}^T Y^*
$$
