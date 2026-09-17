**Part B — Calculation Exercises**

*Exercise 1 — Count Vector*

corpus:
D1 = "cat eats fish"
D2 = "dog eats fish"
D3 = "cat likes fish"

Vocabulary được sắp xếp theo thứ tự:
[cat, dog, eats, fish, likes]


count vector của:
■ (D_1): [1, 0, 1, 1, 0]
■ (D_2): [0, 1, 1, 1, 0]
■ (D_3): [1, 0, 0, 1, 1]

*Exercise 2 — TF*

(D_1): tf(cat, D1) = 1/3

tf(eats, D1) = 1/3

tf(fish, D1) = 1/3

*Exercise 3 — IDF*

df(cat) = 2
df(dog) = 1
df(eats) = 2
df(fish) = 3
df(likes)= 1

idf(cat) $\approx$ 0.176
idf(dog) $\approx$ 0.477
idf(eats) $\approx$ 0.176
idf(fish) = 0
idf(likes) $\approx$ 0.477

*Exercise 4 — TF-IDF*

tfidf(cat, D_1) $\approx$ 0.0587
tfidf(eats, D_1) $\approx$ 0.0587
tfidf(fish, D_1) = 0 

0.8165