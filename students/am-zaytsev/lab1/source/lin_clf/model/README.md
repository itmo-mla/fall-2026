### Постановка задачи и функция потерь

**Определение матриц:**
Даны матрицы \(X\), \(A\) и \(Y\) следующих размерностей:

$$
 X = \begin{bmatrix} x_{0,0} & \dots & x_{0,m} \\ \dots & \dots & \dots \\ x_{n,0} & \dots & x_{n,m} \end{bmatrix}_{(n, m)} \quad
 A = \begin{bmatrix} A_{0,0} \\ \dots \\ A_{m,0} \end{bmatrix}_{(m, 1)} \quad
 Y = \begin{bmatrix} Y_{0,0} \\ \dots \\ Y_{n,0} \end{bmatrix}_{(n, 1)} 
$$

**Вспомогательная переменная \(M\):**
$$
 M = X \cdot A \odot Y
$$

**Функция потерь:**
$$
 \frac{1}{n} \sum_{i=0}^{n} (1 - M_i)^2 + \gamma A^T A = 
$$
$$
 f(X, A, Y) = \sum_{i=0}^{n} (1 - (X_i \times A) \cdot Y_i)^2 + \gamma A^T A 
$$

**Вычисление производной:**
$$
 \frac{df}{dA} = \frac{1}{n} \sum_{i=0}^{n} 2(1 - (X_i \times A) \cdot Y_i) \frac{d(1 - X_i \times A \cdot Y_i)}{dA} + \gamma A \odot 
$$
$$
 \frac{d(1 - Y_i(X_i \times A))}{dA} = 0 - Y_i \cdot X_i^T 
$$

$$
 \frac{1}{n} \sum_{i=0}^{n} (2(1 - (X_i \times A) \cdot Y_i) (0 - Y_i \cdot X_i^T)) + \gamma A 
$$

**Приведем к матричному виду**


$$
 \frac{2}{n} \sum (-Y_i \cdot (1 - (X_i \times A) \cdot Y_i) \times X_i^T) + \gamma A =
$$

$$\text{при }
 M = 1 - (X \times A) \cdot Y
$$
$$
 [M_i = 1 - (X_i \times A) \cdot Y_i]
$$


$$
 = \frac{2}{n} \sum (-Y_i \cdot M_i) \times X_i^T + \gamma A= |\text{при } S_i = -Y_i \cdot M_i|= \frac{2}{n} \sum S_i \times  X_i^T + \gamma A= 
$$


$$
 = \frac{2}{n}(S^T \times X)^T + \gamma A=\frac{2}{n} X^T \times S + \gamma A = \frac{2}{n} X^T \times (-Y \cdot (1 - X \times A \cdot Y)) + \gamma A= 
$$
$$
=-\frac{2}{n} X^T \times Y +  \frac{2}{n} X^T \times (Y^2 \cdot X)\times A + \gamma A
$$
$$
=-\frac{2}{n} X^T \times Y + \left( \frac{2}{n} X^T \times (Y^2 \cdot X) + \gamma I \right) \times A
$$
**Вектор градиента:**
$$
 \boxed{ \nabla L(A) = \left[ -\frac{2}{n} X^T \times Y + \left( \frac{2}{n} X^T \times (Y^2 \cdot X) + \gamma I \right) \times A \right]_{(m, 1)} }
$$

$$\text{при }
 C_1 = \left[-\frac{2}{n} X^T \times Y\right]_{(m, 1)} 
$$
$$\text{при }
 C_2 = \left( \frac{2}{n} X^T \times (Y^2 \cdot X) + \gamma I \right)_{(m, m)} 
$$

$$
 \boxed{ \nabla L(A) = \left[ C_1 + C_2\times A \right]_{(m, 1)} }
$$

---

### Поиск поиск оптимального шага



**Поиск шага \(h\):**

$$
 \phi(h) = L(A - h \nabla L(A)) \to \min_h 
$$
$$
 \frac{d\phi}{dh} = 0 
$$
$$
 \frac{d\phi}{dh} = \nabla L(A - h \nabla L(A))^T \times \nabla L(A) = 0 \quad | D = \nabla L(A) 
$$
$$
 \nabla L(A - h D)^T \times D = 0 
$$
$$
 (C_1 + C_2 \times (A - h D))^T \times D = 0 
$$
$$
 D^T \times (C_1 + C_2 \times (A - h D)) = 0 
$$
$$
 D^T \times C_1 + D^T \times C_2 \times A - h \cdot D^T \times C_2 \times D  = 0 
$$


$$
 h \cdot D^T \times C_2 \times D = D^T \times C_1 + D^T \times C_2 \times A 
$$

**Итоговая формула оптимального шага:**
$$
 \boxed{ h^* = \frac{D^T \times C_1 + D^T \times C_2 \times A}{D^T \times C_2 \times D} } 
$$

**Вспомогательные коэффициенты:**
$$
 C_1 = -\frac{2}{n} X^T \times Y 
$$
$$
 C_2 = \frac{2}{n} X^T \times (Y^2 \cdot X) + \gamma I 
$$
$$
 D = C_1 + C_2 \times A 
$$
