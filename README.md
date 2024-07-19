# **Etude de l'influence des indicateurs macro-économiques sur le cours de l'EURO/USD**

## **I. Introduction:**

Ce projet est une initiation à la finance quantitative, visant à explorer l'influence des indicateurs macro-économiques sur le cours de l'EUR/USD. Le taux de change entre l'euro et le dollar est influencé par plusieurs facteurs extérieurs, notamment les politiques monétaires, la situation politique, la santé économique et les conflits dans chaque région. Les indicateurs macro-économiques de la zone euro et des États-Unis, qui reflètent la santé économique de ces zones, jouent un rôle crucial.

Nous partons du postulat que la valeur de l'EUR/USD peut être exprimée comme une somme pondérée de différents indicateurs économiques. Plus précisément, nous supposons que le prix de l'EUR/USD à un temps $ t+T $ (où $ T $ représente un certain horizon temporel) peut être approximé par le prix à l'instant $ t $ additionné d'une somme pondérée des valeurs des indicateurs $ I $ à l'instant $ t $, composée par une certaine fonction $ f $, ce qui est représenté par :


$$
P_{t+T} = P_t + \sum_{i=1}^n w_i f_i(I_i(t)) + \epsilon
$$

où :
- $ P_{t+T} $ est le prix de l'EUR/USD à l'instant $ t+T $, où $T$ représente un certain horizon temporel
- $ P_t $ est le prix de l'EUR/USD à l'instant $ t $,
- $ I_i(t) $ représente la valeur du $ i $-ème indicateur macro-économique à l'instant $ t $,
- $f_i$ représente une fonction de transformation qui peut-être une normalisation, une transformation logarithmique, une différenciation...
- $ w_i $ est le poids associé à l'indicateur $ I_i(t) $,
- $ \epsilon $ est un terme d'erreur.


Si ce postulat est correct, cela permettra de prédire le prix futur de l'EUR/USD, ce qui est utile pour le pricing d'options ou la gestion de portefeuille. L'objectif de cette étude est de vérifier la validité de ce postulat en analysant la relation entre les indicateurs macro-économiques et le taux de change EUR/USD.

Pour cela, nous allons :
1. Constituer un dataset avec le prix de l'EUR/USD et 25 indicateurs macro-économiques clés.
2. Analyser la corrélation entre ces indicateurs et le prix de l'EUR/USD.
3. Développer des modèles de prédiction basés sur ces indicateurs pour estimer le prix futur de l'EUR/USD.
4. Tester et valider ces modèles avec des données historiques réelles.
