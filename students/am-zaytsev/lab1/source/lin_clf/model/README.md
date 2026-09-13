### Обозначения
Для избежания неоднозначности в формулах используются следующие обозначения:
*   $A \cdot B$ или $AB$ — матричное умножение.
*   $A \odot B$ — поэлементное умножение (произведение Адамара).
*   $A^T$ — транспонирование матрицы $A$.
*   $A_i$ — $i$-я строка (или элемент) матрицы/вектора $A$.

---

### Постановка задачи и функция потерь

**Определение матриц:**
Даны матрицы $X$, $A$ и $Y$ следующих размерностей:

$$
X = \begin{bmatrix} x_{0,0} & \dots & x_{0,m} \\ \dots & \dots & \dots \\ x_{n,0} & \dots & x_{n,m} \end{bmatrix}_{(n, m)} \quad
A = \begin{bmatrix} A_{0,0} \\ \dots \\ A_{m,0} \end{bmatrix}_{(m, 1)} \quad
Y = \begin{bmatrix} Y_{0,0} \\ \dots \\ Y_{n,0} \end{bmatrix}_{(n, 1)} 
$$

**Вспомогательная переменная $M$:**
$$
M = X \cdot A \odot Y
$$

**Функция потерь:**
$$
f(X, A, Y) = \sum_{i=0}^{n} \left(1 - (X_i \cdot A) \odot Y_i \right)^2 + \gamma A^T A 
$$

**Вычисление производной:**
$$
\frac{df}{dA} = \frac{1}{n} \sum_{i=0}^{n} 2 \left(1 - (X_i \cdot A) \odot Y_i \right) \frac{d(1 - X_i \cdot A \odot Y_i)}{dA} + \gamma A 
$$
$$
\frac{d(1 - Y_i \odot (X_i \cdot A))}{dA} = 0 - Y_i \cdot X_i^T 
$$
$$
\frac{1}{n} \sum_{i=0}^{n} \left( 2(1 - (X_i \cdot A) \odot Y_i) (0 - Y_i \cdot X_i^T) \right) + \gamma A 
$$

**Приведем к матричному виду:**

$$
\begin{align*}
\frac{df}{dA} &= \frac{2}{n} \sum_{i=0}^{n} \left( -Y_i \cdot \underbrace{(1 - (X_i \cdot A) \odot Y_i)}_{M_i} \right) X_i^T + \gamma A \\[10pt]
&= \frac{2}{n} \sum_{i=0}^{n} \underbrace{(-Y_i \odot M_i)}_{S_i} X_i^T + \gamma A \\[10pt]
&= \frac{2}{n} (S^T \cdot X)^T + \gamma A = \frac{2}{n} X^T \cdot S + \gamma A \\[10pt]
&= \frac{2}{n} X^T \cdot (-Y \odot (1 - X \cdot A \odot Y)) + \gamma A \\[10pt]
&= -\frac{2}{n} X^T \cdot Y + \frac{2}{n} X^T \cdot (Y^{2} \odot X) \cdot A + \gamma A \\[10pt]
&= -\frac{2}{n} X^T \cdot Y + \left( \frac{2}{n} X^T \cdot (Y^{ 2} \odot X) + \gamma I \right) \cdot A
\end{align*}
$$


**Вектор градиента:**
$$
\boxed{ \nabla L(A) = \left[ -\frac{2}{n} X^T \cdot Y + \left( \frac{2}{n} X^T \cdot (Y^{\odot 2} \odot X) + \gamma I \right) \cdot A \right]_{(m, 1)} }
$$

Обозначим вспомогательные матрицы:
$$
C_1 = \left[-\frac{2}{n} X^T \cdot Y\right]_{(m, 1)} 
$$
$$
C_2 = \left[ \frac{2}{n} X^T \cdot (Y^{\odot 2} \odot X) + \gamma I \right]_{(m, m)} 
$$

Тогда градиент принимает компактный вид:
$$
\boxed{ \nabla L(A) = \left[ C_1 + C_2 \cdot A \right]_{(m, 1)} }
$$

---

### Поиск оптимального шага $h$

**Постановка задачи оптимизации:**
$$
\phi(h) = L(A - h \nabla L(A)) \to \min_h 
$$
$$
\frac{d\phi}{dh} = 0 
$$

**Вычисление производной по шагу $h$:**
Обозначим $D = \nabla L(A)$.
$$
\begin{align*}
\frac{d\phi}{dh} &= \nabla L(A - h D)^T \cdot D = 0 \\[10pt]
&= (C_1 + C_2 \cdot (A - h D))^T \cdot D = 0 \\[10pt]
&= D^T \cdot (C_1 + C_2 \cdot (A - h D)) = 0 \\[10pt]
&= D^T \cdot C_1 + D^T \cdot C_2 \cdot A - h \cdot D^T \cdot C_2 \cdot D = 0 
\end{align*}
$$
Выражаем $h$:
$$
h \cdot D^T \cdot C_2 \cdot D = D^T \cdot C_1 + D^T \cdot C_2 \cdot A 
$$

**Итоговая формула оптимального шага:**
$$
\boxed{ h^* = \frac{D^T \cdot C_1 + D^T \cdot C_2 \cdot A}{D^T \cdot C_2 \cdot D} } 
$$

**Вспомогательные коэффициенты:**
$$
C_1 = -\frac{2}{n} X^T \cdot Y 
$$
$$
C_2 = \frac{2}{n} X^T \cdot (Y^{\odot 2} \odot X) + \gamma I 
$$
$$
D = C_1 + C_2 \cdot A 
$$