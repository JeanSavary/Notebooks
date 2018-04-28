
# Étude de cas Evaneos
### ( Évaluation de la pertinence d'un A/B test )
-----------------------------------------------

<h2><u> Contexte :</h2>
<div align="justify">
        <p>Après avoir réalisé deux versions d'un même site web, l'une étant la version originale (A) du site et l'autre une version implémentant une nouvelle fonctionnalité (B), nous cherchons à déterminer l'efficacité de la nouvelle fonctionnalité.</p>
        <p>Une version sera considérée plus efficace que l'autre si le taux de conversion y est supérieur, <em> i.e </em> <strong>ratio</strong> entre le nombre de sessions ouvertes sur le site et le nombre de devis demandés.</p>
    <p> Après 1 mois d'essai, les résultats de l'étude sont les suivants : </p>
    <ul>
        <li> La version A a eu 4000 sessions et 200 conversions (taux de conversion de <strong>5%</strong>).</li>
        <li> La version B a eu 4000 sessions et 260 conversions (taux de conversion de <strong>6.5%</strong>). </li>

</div> 

<h2><u> Question : </u></h2>
<div align="justify">
    <p> Ce résultat est-il statistiquement significatif , <em>i.e</em> la différence constatée entre les deux versions est-elle <strong>uniquement</strong> due à la nouvelle fonctionnalité ?</p>
</div>

----------------------

<h2> Qu'est-ce qu'un test A/B ?</h2>
<div align="justify">
    <p> Un test A/B est un type de test permettant de comparer l'efficacité de deux versions d'un même site quant à la réalisation d'un objectif (ici augmenter le taux de conversion). Dans notre cas nous avons nos deux versions du site : la version <strong>A</strong> (originale) et la version <strong>B</strong> (implémentant la nouvelle fonctionnalité). </p>
    <p> Lors d'un A/B test il faut <strong> absolument </strong> comparer les deux candidats sur la même base, <em>i.e</em> les deux versions doivent être soumises au test dans les <strong> mêmes </strong> conditions.</p>
    <p> Nos résultats montrent que le test B a de meilleurs résultats que le test A, néanmoins on ne sait pas si c'est essentiellement lié à la nouvelle fonctionnalité, cela peut être dû à différents facteurs, notamment le <strong>hasard</strong>. Il faut par conséquent prouver que ces résultats sont <strong> statistiquement significatifs</strong>.</p>
</div>

--------

<h2> Étude statistique : </h2>
<div align="justify">
    <p> Comme évoqué précédemment, nous allons <strong>modéliser</strong> le mois d'essai en utilisant cette fois les statistiques de manière à confirmer ou non la pertinence des résultats obtenus. Cette méthode est essentielle car elle permettra de fournir un résultat général (en augmentant le nombre de sessions par exemple) que nous comparerons ensuite avec le résultat réel. </p>
    <p> Pour se faire nous avons besoin de créer un modèle statistique qui soit le plus proche de la réalité. Nous avons dans le cas présent, des internautes qui visitent le site et demandent un devis avec une probabilité d'approximativement <strong>7%</strong>.</p>
    <p> La <strong>loi binomiale</strong> est une modélisation mathématique se rapprochant de la réalité dans notre cas. Effectivement, celle-ci représente le <strong>nombre de succès</strong> (ici demande d'un devis) que l'on obtient en réalisant des expériences <strong>identiques, aléatoires et indépendantes</strong> entre elles. Nous ferons donc l'hypothèse que chaque internaute visitant le site est indépendant des autres internautes (cela signifie que chaque utilisateur visite le site de sa propre initiative, personne ne l'a influencé à faire cela). Nous supposerons qu'1 utilisateur = 1 session. De plus, le site étant identique pour chaque individu, les expériences utilisateurs seront donc identiques entre elles, et aléatoires car l'on ne sait pas <em> à priori </em> si l'utilisateur demandera un devis.</p>


<h3><u> Première modélisation :</u> <br /></h3> 
<div align="justify">
    <p> <strong> Hypothèses </strong> : </p>
    <ul>
        <li> La page possède <strong>5000 sessions</strong> (5000 utilisateurs indépendants).</li>
        <li> Le taux de conversion moyen du site est de <strong>7%.</strong> </li>
    </ul><br />
    <p><strong> Simulations du nombre de conversions pour 5000 sessions, avec une probabilité de conversion de 0.07 :</strong></p>
</div>



```python
import matplotlib.pyplot as plt
import numpy as np

#Dans cette fonction nous utiliserons la fonction numpy.random.binomial() qui calcule directement le résultat de n expériences de bernouilli avec probabilité de succès égale à p
def simulation_np (n, p) :
    return (np.random.binomial(n,p))

#Affichage des résultats pour 5 expériences
print("Résultats de 5 simulations : \n")
somme = 0

for i in range(5):
    resultat = simulation_np(5000,0.07)
    somme += resultat
    print("%(result)d conversions parmis 5000 sessions" %{"result" : resultat})

#Calcul de la moyenne de nos 5 expériences
moyenne = somme / 5      
print("\nLa moyenne du nombre de conversions pour nos 5 expériences est de %(moyenne)d conversions" %{"moyenne":moyenne}) 
```

    Résultats de 5 simulations : 
    
    358 conversions parmis 5000 sessions
    347 conversions parmis 5000 sessions
    339 conversions parmis 5000 sessions
    337 conversions parmis 5000 sessions
    367 conversions parmis 5000 sessions
    
    La moyenne du nombre de conversions pour nos 5 expériences est de 349 conversions


----------


```python
#Code alternatif plus concis

#Résultats de nos 5 expériences :
values_ = np.random.binomial(5000,0.07,5)
print("Résultats de 5 simulations : \n")
for value in values_ : 
    print("%(result)d conversions parmis 5000 sessions" %{"result" : value})

#Moyenne de nos 5 expériences
print("\nLa moyenne du nombre de conversions pour nos 5 expériences est de %(moyenne)f conversions" %{"moyenne":np.mean(values_)}) 
```

    Résultats de 5 simulations : 
    
    360 conversions parmis 5000 sessions
    336 conversions parmis 5000 sessions
    338 conversions parmis 5000 sessions
    340 conversions parmis 5000 sessions
    343 conversions parmis 5000 sessions
    
    La moyenne du nombre de conversions pour nos 5 expériences est de 343.400000 conversions


-------

<div align="justify">
    <p> Nous voyons donc que le nombre de conversions varie d'une expérience à l'autre. </p>
    <p> Évidemment la moyenne que nous avons calculé n'est pas très pertinente car le nombre d'expériences réalisées n'est pas suffisamment grand. Ce résultat n'est donc pas <strong>statistiquement significatif</strong> ! </p>
    <p> Réalisons donc <strong>10 000 fois</strong> cette expérience plutôt que 5 fois et voyons ce que ça nous donne ! </p><br />
</div>
<p><strong> Représentation de la loi Binomiale pour 10 000 expériences : </strong></p>


```python
import math
import scipy.stats

#Enregistrement de nos nombres de conversions pour nos 10 000 expériences
#Le troisième paramètre de random.binomial() spécifie le nombre d'expériences que nous voulons réaliser, ici 10 000
valeurs = np.random.binomial(5000,0.07, 10000) 

#Calcul de la moyenne pour nos 10 000 expériences
moyenne = np.mean(valeurs) 

#Calcul de la variance pour nos 10 000 expériences
variance = np.var(valeurs)

#Calcul le nombre de valeurs de conversions différentes lors de la réalisation de nos 10000 expériences
nb_unique_valeurs= len(set(valeurs))

#Représentons désormais l'histogramme de nos valeurs enregistrées
#L'axe des ordonnées de notre histogramme sera normalisé, ie les valeurs prises seront entre 0 et 1
count, bins, ignored = plt.hist(valeurs, bins = nb_unique_valeurs, normed = 1, label='histogramme')
plt.title("Distribution de notre loi binomial, avec n = 5 000 et p = 0.07, pour 10 000 réalisations")
plt.xlabel("Nombre de conversions")
plt.ylabel("Nombre d'occurences normalisé entre 0 et 1")

#Moyenne de notre distribution
print("\nLa moyenne du nombre de conversions pour nos 10 000 expériences est de : %(moyenne)f conversions" %{"moyenne":moyenne})

#Variance de notre distribution
print("La variance de notre distribution est de : %(variance)f" %{"variance" : variance})

#---------------------------
#Loi normale correspondante

#Calcul de l'espérance de notre loi binomiale précédente : mu = n * p
mu = 5000 * 0.07

#Calcul de l'écart-type de notre loi binomiale précédente : sigma = sqrt(variance) = sqrt(n*p*(1-p))
sigma = math.sqrt(mu* (1-0.07))

plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=1, color='r', label='densite de probabilite loi normale')
plt.axvline(x=375,color='g',label='seuil de significativité', linewidth = 1)
plt.legend()
plt.show()

print("\nEspérance de la loi binomiale : %(esp)f" %{"esp" : mu})
print("Écart-type de la loi normale : %(sigma)f " %{"sigma" : sigma})
```

    
    La moyenne du nombre de conversions pour nos 10 000 expériences est de : 350.318600 conversions
    La variance de notre distribution est de : 321.819894



![png](output_12_1.png)


    
    Espérance de la loi binomiale : 350.000000
    Écart-type de la loi normale : 18.041619 


<div align="justify">
    <p> On remarque alors que notre loi binomiale ressemble à une <strong>distribution normale </strong> (courbe en <strong>rouge</strong>) de moyenne et écart-type mentionnés en dessous de notre figure.</p>
    <p> Cette fois-ci, même si la moyenne calculée est proche de celle que l'on avait précédemment calculé (avec 5 expériences), la moyenne a du sens "statistiquement" car elle est calculée à partir d'une grande population.</p> 
    <p> Cette nouvelle approximation nous permettra de réaliser des calculs de probabilité plus facilement, voyons donc cela !</p>
</div>

-----

<p><strong>Calculons désormais la probabilité d'obtenir un nombre de conversions supérieur ou égal à 375 :</strong></p>


<p> Nous utiliserons donc les <strong>formules</strong> de calcul de probabilité que nous offrent les lois normales.</p>
<p>Nous cherchons à calculer la probabilité suivante :</p>
\begin{equation*}
P(X\geq 375)
\end{equation*}
<div align="right">
<p><em>Où X est notre nombre de conversions.</em></p>
</div>


<p> Il n'y a pas de formule directe pour calculer cette probabilité, néanmoins nous connaissons la formule pour calculer :<br /></p>
\begin{equation*}
P(X \leq 374)
\end{equation*}
<p> Nous obtiendrons ensuite le résultat grâce à :</p>
\begin{equation*}
P(X\geq 375) = 1 - P(X\leq 374)
\end{equation*}

<p> Or, ce que nous cherchons à calculer (la fonction de répartition de la loi normale) correspond à <strong>l'intégrale de sa densité de probabilité</strong>. Nous obtenons donc l'égalité suivante : </p>


\begin{equation*}
P(X \leq 374) = \int_{-\infty}^{374} \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2}(\frac{x-m}{\sigma})^2) \, \mathrm{d}x \
\end{equation*}
<p><em>, avec m la moyenne de notre loi normale et sigma l'écart-type</em></p>
<p><strong>Calculons donc cette valeur de probabilité à l'aide de python </strong>:</p>



```python
#La librairie scipy nous fournit la fonction stats.norm.cdf(x,m,sigma) qui donne directement la probabilité que X<=374
proba = scipy.stats.norm.cdf(374,mu,sigma)

#Nous trouvons ensuite la valeur désire 1 - P(X<=374)
resultat = 1 - proba
print("La probabilité de trouver un nombre de conversions supérieur ou égal à 375 vaut : %(resultat)f" %{"resultat" : resultat})
print("Nous avons donc %(pourcentage)f%%  de chance d'obtenir un nombre de conversions supérieur ou égal à 375 !" %{"pourcentage" : resultat*100})

#On aurait pu calculer ce résultat directement grâce à la fonction scipy.stats.norm.sf()
print("\nProbabilité obtenue par calcul direct : %(proba)f" %{"proba" : scipy.stats.norm.sf(374,mu,sigma)})
```

    La probabilité de trouver un nombre de conversions supérieur ou égal à 375 vaut : 0.091717
    Nous avons donc 9.171671%  de chance d'obtenir un nombre de conversions supérieur ou égal à 375 !
    
    Probabilité obtenue par calcul direct : 0.091717


<p><strong> Pourquoi avoir calculé cette probabilité ? </strong></p>
<p> Nous avons calculé cette probabilité car un taux de conversion est considéré comme étant <strong>anormalement élevé</strong>, <em>i.e</em> que la différence entre les deux versions n'est pas due au hasard mais bien à l'efficacité de la nouvelle fonctionnalité, lorsque la probabilité précédemment calculée est inférieure à <strong>5%</strong>. Ainsi dans notre exemple de modélisation (5000 sessions, taux moyen de 7%), le taux n'est <strong>pas</strong> anormalement élevé (9% > 5%).</p>

--------

<h3><u> Deuxième modélisation :</u> <br /></h3> 
<div align="justify">
    <p> <strong> Hypothèses </strong> : </p>
    <ul>
        <li> La page possède <strong>10 000 sessions</strong> (10 000 utilisateurs indépendants).</li>
        <li> Le taux de conversion moyen du site est de <strong>7%.</strong> </li>
    </ul><br />
    <p><strong> Réitérons les étapes précédentes pour obtenir la probabilité d'avoir un nombre de conversions supérieur ou égal à 750 :</strong></p>
</div>

<p><u>1) Obtention de <strong>l'histogramme</strong> de notre loi binomiale :</u></p> 


```python
#On génère les nombres de conversions obtenus lors de nos 10 000 essais
valeurs2 = np.random.binomial(10000,0.07,10000)

#On récupère la moyenne, variance, nombre de valeurs différentes
moyenne2 = np.mean(valeurs2)
variance2 = np.var(valeurs2)
nb_unique_valeurs2 = len(set(valeurs2))

#On représente l'histogramme de notre loi binomiale
count2, bins2, ignored2 = plt.hist(valeurs2, bins = nb_unique_valeurs2, normed = 1, label="histogramme")
plt.title("Distribution de notre loi binomiale, avec n = 10000 et p = 0.07, pour 10 000 réalisations")
plt.xlabel("Nombre de conversions")
plt.ylabel("Nombre d'occurences normalisé entre 0 et 1")

#Moyenne de notre distribution
print("\nLa moyenne du nombre de conversions pour nos 10 000 expériences est de : %(moyenne)f conversions" %{"moyenne":moyenne2})

#Variance de notre distribution
print("La variance de notre distribution est de : %(variance)f" %{"variance" : variance2})

#---------------------------
#Loi normale correspondante

#Calcul de l'espérance de notre loi binomiale précédente : mu = n * p
mu2 = 10000 * 0.07

#Calcul de l'écart type de notre loi binomiale précédente : sigma = sqrt(variance) = sqrt(n*p*(1-p))
sigma2 = math.sqrt(mu2 * (1-0.07))

plt.plot(bins2, 1/(sigma2 * np.sqrt(2 * np.pi)) * np.exp( - (bins2 - mu2)**2 / (2 * sigma2**2) ), linewidth=1, color='r', label='densité de probabilité loi normale')
plt.legend()
plt.show()

print("\nEspérance de la loi binomiale : %(esp)f" %{"esp" : mu2})
print("Écart-type de la loi normale : %(sigma)f " %{"sigma" : sigma2})
```

    
    La moyenne du nombre de conversions pour nos 10 000 expériences est de : 700.100800 conversions
    La variance de notre distribution est de : 665.873839



![png](output_25_1.png)


    
    Espérance de la loi binomiale : 700.000000
    Écart-type de la loi normale : 25.514702 


-----

<p><u>2) Calcul de la probabilité d'obtenir un nombre de conversions <strong>supérieur ou égal à 750</strong> :</u></p>


<p>Cette fois-ci on cherche la probabilité suivante : </p>
\begin{equation*}
P(X\geq 750)
\end{equation*}
<p>Soit le calcul suivant : </p>
\begin{equation*}
P(X \leq 749) = \int_{-\infty}^{749} \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2}(\frac{x-m}{\sigma})^2) \, \mathrm{d}x \
\end{equation*}
<p>Et toujours : </p>
\begin{equation*}
P(X\geq 750) = 1 - P(X\leq 749)
\end{equation*}
<p><strong>Calculons cette valeur à l'aide de python :</strong></p>


```python
#La probabilité que X<=749
proba2 = scipy.stats.norm.cdf(749,mu2,sigma2)

#Nous trouvons ensuite la valeur désire 1 - P(X<=749)
resultat2 = 1 - proba2
print("La probabilité de trouver un nombre de conversions supérieur ou égal à 750 vaut : %(resultat)f" %{"resultat" : resultat2})
print("Nous avons donc %(pourcentage)f%%  de chance d'obtenir un nombre de conversions supérieur ou égal à 750 !" %{"pourcentage" : resultat2*100})


```

    La probabilité de trouver un nombre de conversions supérieur ou égal à 750 vaut : 0.027400
    Nous avons donc 2.739982%  de chance d'obtenir un nombre de conversions supérieur ou égal à 750 !


----

<div align="justify">
<p>Cette fois-ci nous avons donc une probabilité d'avoir un taux de conversion réel supérieur à 7.5% de : <strong>2.74%</strong>.</p>
<p>Nous sommes donc <strong>sous la limite</strong> des 5% ! Ce qui signifie maintenant que nous avons un taux de conversion <strong>anormalement élevé</strong>, synonyme potentiellement de l'efficacité d'une nouvelle fonctionnalité.<br/></p>
<p>Ce résultat illustre une <strong>notion importante</strong> ! En effet, pour une <strong>même</strong> technologie, <em>i.e</em> un site détenant un taux de conversion de <strong>7%</strong>, le nombre de sessions à considérer est un <strong>facteur clé</strong> quant à la détermination de l'efficience d'une nouvelle fonctionnalité. Dans notre premier cas, nous n'avions considéré que <strong>5 000 sessions</strong>, ce qui nous aurait conduit à conclure que la nouvelle fonctionnalité n'était pas efficace si nous n'avions pas essayé avec <strong>10 000 sessions</strong>. Cela aurait conduit potentiellement à une <strong>perte de chiffre d'affaire</strong> suite à une erreur d'analyse ! Il <strong>faut</strong> tenir compte du plein potentiel du site : si l'on considère 5 000 sessions alors que le site possède un nombre moyen de 10 000 sessions, alors on ne réalise pas l'étude statistique sur l'environnement réel.</p>
<p>On pourrait alors se demander jusqu'à quel point on peut augmenter le nombre de sessions considérées ? Selon moi, il faut considérer le nombre de sessions des <strong>années passées à la même saison</strong> (la demande varie selon les saisons) ainsi que le nombre moyen de sessions des <strong>précédents mois</strong> (pour tenir compte de l'essort du site). On choisira alors un nombre de sessions fidèle aux données récoltées précédemment.<br /></p>
<p><strong><u>Edit :</u></strong> J'ai lu diverses documentations sur les tests statistiques, et j'ai pris connaissance de la notion de <strong>"puissance statistique"</strong>. Celle-ci correspond à la probabilité de détecter une différence entre deux versions sachant que cette différence existe vraiment. On cherche donc naturellement à avoir une <strong>forte</strong> puissance statistique (généralement supérieure à <strong>0.8</strong>). Ainsi, si notre puissance statistique est trop faible, on augmente le risque de <strong>ne pas détecter</strong> de différence entre 2 versions alors qu'il en existe une ! Or, parmis les facteurs influençant la puissance statistique, se trouve la <strong>taille de l'échantillon</strong> que l'on considère dans notre étude (ici le nombre de sessions). On comprend donc pourquoi, avec 5 000 sessions on ne trouvait pas le même résultat qu'avec 10 000 sessions (notre puissance était <strong>trop faible</strong>). Pour en revenir à la question de la taille d'échantillon, il est possible de calculer via une formule connue, le <strong>nombre minimum</strong> d'échantillons pour garantir une puissance donnée (0.8 par exemple).</p>
</div>

----

<p><strong>Représentation du seuil de significativité statistique sur histogramme (10 000 sessions) : </strong></p>


```python
#Représentation de l'histogramme précédent
plt.hist(valeurs2, bins = nb_unique_valeurs2, normed = 1)
plt.title("Distribution de notre loi binomiale, avec n = 10000 et p = 0.07, pour 10 000 réalisations")
plt.xlabel("Nombre de conversions")
plt.ylabel("Nombre d'occurences normalisé entre 0 et 1")

#Représentation du seuil de significativé à 750
plt.axvline(x=750,linewidth=1,color='r', label="Seuil de significativité")

plt.legend()
plt.show()
```


![png](output_34_0.png)


----

<h3> Réponse à la question initiale :</h3>
<p> On se place sous hypothèse nulle, <em>i.e</em> dans le cas de la version A, et nous allons calculer la probabilité d'obtenir un taux de conversion supérieur ou égal au taux moyen obtenu sur la version B qui est de <strong>6.5%</strong>(soit 260 conversions).</p>
<p> 1) Commençons comme toujours par simuler l'expérience réalisée sur la version A par une loi binomiale de paramètre n = 4 000 et p = 0.05.</p>
<p> 2) Représentons alors cette distribution via un <strong>histogramme</strong>, sur lequel on représente l'allure de la <strong>densité de probabilité</strong> de la loi normale correspondante. Le <strong>seuil</strong> de significativité statistique a lui aussi été déssiné en vert sur le graphe.</p>


```python
#Considérons qu'on réalise 10 000 fois l'expérience avec 4 000 utilisateurs et un taux de conversion moyen de 6.5% (cas version B)
#Calculons les valeurs clés du problème
valeurs3 = np.random.binomial(4000,0.05,10000)
moyenne3 = np.mean(valeurs3)
variance3 = np.var(valeurs3)
nb_unique_valeurs3 = len(set(valeurs3))

mu3 = 4000 * 0.05
sigma3 = math.sqrt(mu3*(1-0.05))

#Traçons l'histogramme et la densité de probabilité de la loi normale correspondante
count3, bins3, ignored3 = plt.hist(valeurs3, bins = nb_unique_valeurs3, normed = 1, label="histogramme")
plt.plot(bins3, 1/(sigma3 * np.sqrt(2 * np.pi)) * np.exp( - (bins3 - mu3)**2 / (2 * sigma3**2) ), linewidth=1, color='r', label='densité de probabilité loi normale')
plt.axvline(x=260, color='g', label='seuil de significativité', linewidth=1)
plt.title("Distribution de notre loi binomiale, avec n = 4 000 et p = 0.05, pour 10 000 réalisations")
plt.xlabel("Nombre de conversions")
plt.ylabel("Nombre d'occurences normalisé entre 0 et 1")
plt.legend()
plt.show()

print("\nMoyenne de la loi normale : %(esp)f" %{"esp" : mu3})
print("Écart-type de la loi normale : %(sigma)f " %{"sigma" : sigma3})
```


![png](output_37_0.png)


    
    Moyenne de la loi normale : 200.000000
    Écart-type de la loi normale : 13.784049 


<p><strong>3) On cherche à caculer la probabilité suivante :</strong> </p>
\begin{equation*}
P(X\geq 260) = 1 - P(X\leq 259)
\end{equation*}
<p> avec : </p> 
\begin{equation*}
P(X \leq 259) = \int_{-\infty}^{259} \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2}(\frac{x-m}{\sigma})^2) \, \mathrm{d}x \
\end{equation*}


```python
print("Avec la version A du site, nous avons %(pourcentage)f%%  de chance d'obtenir un nombre de conversions supérieur ou égal à 260 !" %{"pourcentage" : scipy.stats.norm.sf(259,mu3,sigma3)})



```

    Avec la version A du site, nous avons 0.000009%  de chance d'obtenir un nombre de conversions supérieur ou égal à 260 !


<p>La probabilité calculée est alors <strong>très inférieure</strong> à la limite de 5%. Ceci était d'ailleurs prévisible en observant le placement du seuil de significativité statistique sur notre histogramme.</p>

<p>Il est important de noter qu'en cas de taille d'échantillonage trop faible, nous pouvons être en situation d'une importante dose de faux-positifs, <em>i.e</em> nous allons conclure que le site B est plus efficace que le site A alors que ce n'est pas le cas.</p> 
<p>Par précaution, réalisons donc la même procédure mais avec des sessions de 8 000 (8 000 utilisateurs est une donnée plausible quant à la fréquentation du site, car si il n'y avait qu'une seule version du site, alors on aurait eu 4000 + 4000 utilisateurs).</p>


```python
#Calculs directs sans graphique
valeurs4 = np.random.binomial(8000,0.05,10000)
mu4 = 8000*0.05
sigma4 = math.sqrt(mu4 * (1-0.05))

#Cette fois nous cherchons P(X >= 520) car un taux de 6.5% pour 8000 sessions donne 520 conversions 
proba4 = scipy.stats.norm.sf(519,mu4,sigma4)
print("Dans la version A du site, avec 8 000 sessions, nous avons %(proba)f%% de chance d'obtenir un nombre de sessions supérieur ou égal à 520." %{"proba" : proba4 * 100})

```

    Dans la version A du site, avec 8 000 sessions, nous avons 0.000000% de chance d'obtenir un nombre de sessions supérieur ou égal à 520


<p>Le résultat précédent confirme alors notre l'hypothèse selon laquelle <strong>la version B est plus efficace que la version A et ce essentiellement grâce à la nouvelle fonctionnalité</strong>.</p>
<p><strong><u>NB :</u></strong> Néanmoins, je n'ai pas ici calculé la puissance statistique de notre modèle car je ne suis pas sûr de la méthode à employer. Or c'est un facteur essentiel quant à la conclusion du problème. Ainsi, si la puissance de notre modèle est supérieur à 0.8 alors notre conclusion est définitivement vraie. Dans le cas contraire, la conclusion a de grandes chances d'être fausse...</p>

----

<h3><u> Question subsidiaire :</u></h3>
<p><strong> Plutôt que de mettre en ligne les deux versions en même temps, un membre de l'équipe propose de mettre d'abord en ligne la version A, puis de mettre en ligne la version B. Qu'en pensez-vous?</strong></p>

<div align="justify">
<p> Comme mentionné dans l'introduction, lors de la réalisation d'un test A/B, les deux versions doivent absolument être soumises aux <strong>mêmes conditions </strong> (même environnement, <em>i.e</em> toutes les variables extérieures à notre problème doivent être identiques). Lorsque l'on compare 2 sites en parallèle, les deux sites fonctionnent dans un <strong>même environnement </strong>: même période, les utilisateurs sont considérés indépendants entre eux, etc...</p>
<p>Néanmoins, si l'on met en ligne le site A (1 mois par exemple) et qu'après 1 mois on mette en ligne le site B, alors on <strong>n'est pas assuré que les deux sites fonctionnent sous les mêmes conditions</strong>. Pour cela, il faudrait par exemple étudier les données des <strong>années passées</strong> pour choisir 2 mois successifs dont l'activité est relativement <strong>constante</strong> (<em>i.e</em> nombre constant de sesions par exemple). Cependant, l'un des avantages de cette méthode, est que l'on est <em>à priori</em> garanti de ne pas avoir de problème de <strong>taille d'échantillon trop faible</strong>, comme cela aurait pu se passer lors de la mise en parallèle des deux sites.</p>
    <p>À mon idée, si l'entreprise peut garantir des <strong>conditions identiques</strong> pour les <strong>deux</strong> sites mis en ligne de manière séquentielle alors il est préférable de le faire. Dans le cas contraire, il est nécessaire de rester sur une mise en ligne parallèle.</p>
</div>
