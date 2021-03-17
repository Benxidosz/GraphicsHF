# Számítógépes Grafika Skeleton Linux felhasználók számára

### Disclaimer: Ez a megoldás a Tanár Úrtól függetlenül, hallgatók által jött létre. Használat előtt értsd meg, hogy mit csinálsz a rendszereddel, hogyan áll össze az egész. Ha valami nem világos a builddel, nyugodtan keress meg, és a technikai részletek setupolásában szívesen segítek. Ettől függetlenül természetesen a házi feladat legfontosabb eleme az önálló megoldás, így kérlek ne kérj közvetlen segítséget a megoldásával kapcsolatban.

### 2. Disclaimer: A fenti azt is jelenti, hogy semmilyen garanciát nem tudok vállalni arra, hogy a keretrendszer fájljai naprakészek a tárgyoldalon találhatóhoz képest. Éppen ezért, mielőtt nekilátsz a házifeladatnak, ezeket kérlek frissítsd magadnál a tárgyoldal alapján. *(Egyszerűen másold át az új framework.h és framework.cpp fájlokat.)*

## Függőségek telepítése - avagy fegyverkezzünk fel a csatára

A fordításhoz több szoftvercsomagra (package-re) is szükségünk lesz. Ezeket
a tárgyoldalon elérhető Móricka keret a programokhoz Visual Studio projekt a lib
mappában direkt tartalmazza. Linuxon azonban egyszerűbb megoldás az operációs
rendszerünk csomagkezelőjével feltelepíteni ezeket. Az Ubuntu csomagkezelője
például az apt. Töltsünk is le ezzel mindent, ami szükséges a programunk
fordításához.

```
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libglew-dev
sudo apt-get install freeglut3-dev
```

## Buildelés - történetünk bonyodalma és tetőpontja

A cmake program segítségével fogjuk generálni a Makefileunkat, ami a programunk
buildeléséhez tartalmaz információkat.

Hogy ne a munkamappát szemetelje tele a buildelés, hozz létre egy `build`
mappát, majd lépj bele.

```
mkdir build
cd build
```

Mivel a build mappa ősében található a `CMakeLists.txt` fájl, adjuk meg az
eggyel fentebb lévő mappát a cmake programnak *(a `build` mappából)*:

```
cmake ..
```

Ezen a ponton egy csomó fájl jött létre a `build` mappában. *(Ezekkel nem akartuk
tele szemetelni a munkamappát.)* Ezekből nekünk a legfontosabb a `Makefile` —
ez tartalmazza a programunk buildeléséhez szükséges információkat.

Adjuk ki a `make` parancsot a program buildeléséhez:

```
make
```

## A Program Futtatása - avagy arassuk le a babérokat

Ekkor már ténylegesen lefordult a programunk, a futtatható bináris fájlunk el is
készült `Skeleton` néven. Ezt futtassuk, és csodáljuk meg, hogy milyen gyönyörű
is a grafika. :)

```
./Skeleton
```

## Ha elakadnál

Ha bármelyik ponton nem sikerülne a mintaprogram létrehozása, nyugodtan írj egy
e-mailt a [barnabas.borcsok@gmail.com](mailto:barnabas.borcsok@gmail.com) címre,
hogy ne a technikai részletek miatt ne tudd kiélvezni a házi feladat önálló
elkészítésének varázslatos élményét.

## Kényelmi script a buildeléshez fejlesztés közben

A `buildrun.sh` shell script a fent leírt lépéseket automatizálja. Nem muszáj
használni, de kényelmes lehet fejlesztés közben nyitvahagyni egy terminál
ablakot, ahol a `./buildrun.sh` paranccsal futtatva a script lebuildeli
a programunkat a `build` mappába, futtatja a lefordított programot, majd
visszalép a projekt gyökérmappájába.

## A kódbázis letöltése - ha nem lennél ismerős a git világában

A fenti git repót könnyedén le tudod klónozni a gépedre a szokásos módon:
```
git clone git@git.sch.bme.hu:bobarna/grafika-linux-starter-pack.git
```

Ha azonban nem szeretnél gitet használni, akkor is könnyedén le tudod tölteni a kódbázist zip fájlként az oldal tetején.

![A kódbázis letöltése zip fájlként](https://i.imgur.com/c1n7iKS.png)
