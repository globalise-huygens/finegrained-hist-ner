# NER Guidelines
> created: 19.07.2023

> revised: 20.11.2023; 24.06.2024

> last updated: 24.06.2024

## General Directions
- break down components of compositional entities: in *kirmanse wol*, annotate *kirmanse* and *wol* separately. In this round of annotations we only consider surface-level annotations; full entity spans will be addressed later, when we annotate entities for event detection and entity linking. The token-based annotations that we collect now will serve as input for span-based detection of entities. 
- Entities should receive a single label, based on the surface form of the entity: *persien* should be annotated as `LOC_NAME`, and not both as `LOC_NAME` and `ORG`; the link to polities will be addressed when doing entity linking.
- When annotating do not include trailing punctuation or articles (except for "„" and "." in commodity quantities, and abbreviation indicators). Only annotate those tokens which fit the guide for that type of entity (See [Entity Classes](#entity-classes)). 
  - articles: *den [8en december 1687]<sub>DATE</sub>* NOT *<font color=red>[den 8en december 1687]<sub>DATE</sub></font>*
  - abbreviation: *[6 novemb:]<sub>DATE</sub> aengekomen*
  - punctuation: *[syde]<sub>CMTY_NAME</sub>, [clede]<sub>CMTY_NAME</sub>, [poyersuycker]<sub>CMTY_NAME</sub>*
- Pay attention to pre-annotations and correct these where needed (See [Pre-Annotations](#pre-annotations))
- Annotations are constrained to agree with tokens, as identified by a tokenizer. Annotate the full span when it expands beyond the actual tokens. The following should all be annotated as a whole.
  - *dEngelse* 
  - *t'Jacht*
  - *CCoromandel*
- Do not annotate entities which have become unrecognizable from HTR errors. Small errors should be ignored.
  - Do not annotate "„ 32 „ 4" (an HTR transcription of Japan) as being a location.
- Be careful to distinguish CMTY_QUANT and CMTY_QUAL

<!-- <span style="background-color: #FFFF00">This text is highlighted in yellow.</span> -->
<!-- <font color=red>text</font> -->

## Difficult terms
These documents include various difficult or foreign terms. For unfamiliar terms use the existing thesauri, ask in the `#globalise-annotation` slack-channel or add them to the [spreadsheet]().

If the above options do not resolve the issue and you are still unsure about the class you can leave the span unclassified. Select the span but do not assign a type. This flags the span for the curator or for later group-discussion. Do try to resolve types as much as possible.

## Pre-Annotations

The documents have been pre-annotated. Pre-annotations may be incorrect. Correct or remove incorrect pre-annotations.

1. Identify whether the span should be annotated at all.
    <details>
      <summary><i>dat zig <span style="background-color: #FFCCCB">[Inde]<sub>LOC_NAME</sub></span> rivieren van [pamalang]<sub>LOC_NAME</sub>, [soumor]<sub>LOC_NAME</sub>, [paccalongan]<sub>LOC_NAME</sub> en [Mabehoer]<sub>LOC_NAME</sub></i></summary>
      <font color=gray><br><i>Inde</i> should not be annotated at all.</font>
    </details>   
2. Identify whether the span has the correct NER label.
    <details>
      <summary><i>Van <span style="background-color: #FFCCCB">[Cheribon]<sub>PER_NAME</sub></span> 6 Maij</i></summary>
      <font color=gray><br><i>Cheribon</i> should be LOC_NAME not PER_NAME</font>
    </details>
3. Identify whether the span is correct.
    <details>
      <summary><i>genoemde <span style="background-color: #FFCCCB">[Cap.]<sub>PRF</sub></span> n voor Tolk</i></summary>
      <font color=gray><br>The class (PRF) is correct but the span should include the entire term: <i>[Cap.n]<sub>PRF</sub></i></font>
    </details>
    <details>
      <summary><i><span style="background-color: #FFCCCB">[Commis. s tak]<sub>PRF</sub></span></i></summary>
      <font color=gray><br>In this case the span is too long: <i>[Commis. s]<sub>PRF</sub></i></font>
    </details>
    <details>
      <summary><i>De [schepen]<sub>SHI_TYPE_</sub> <span style="background-color: #FFCCCB">[prins]<sub>SHIP</sub></span> <span style="background-color: #FFCCCB">[Willem]<sub>PER_NAME</sub></span> <span style="background-color: #FFCCCB">[Hendrik]<sub>SHIP</sub></span></i></summary>
      <font color=gray><br>In this case the span has been needlessly split up. <i>de [schepen]<sub>SHIP_TYPE</sub> [prins Willem Hendrik]<sub>SHIP</sub></i></font>
    </details>
4. Pay special attention to terms you are not familiar with.
    <details>
      <summary><i><span style="background-color: #FFCCCB">[Cannaser]<sub>LOC_ADJ</sub></span></i></summary>
      <font color=gray><br>Cannasser is not a LOC_ADJ it is a measure of capacity so likely CMTY_QUANT</font>
    </details>
5. Do not let pre-annotations blind you from un-annotated terms particularly in lists.
    <details>
      <summary><i>1. sergiant 1. [Chirurgijn]<sub>SHIP</sub> en 10. zoldaten</i></summary>
      <font color=gray><br>Not every span that should be annotated has been pre-annotated. Especially when there are a lot of pre-annotations it can be easy to overlook unannotated terms that should be annotated. <br> In this case, only Chirurgijn has been pre-annotated (this should be corrected to PRF). While not pre-annotated sergiant and zoldaten should also be annotated, both as PRF.</font>
    </details>


## Entity Classes

| NER label | Description | Example | Related entities
| --------- | ----------- | ------- | ---------------------
PER_NAME | Name of Person | Tilling | persons
PRF | Profession, title | Conincx   | persons
STATUS | (civic) status | veduwe, slaaf | persons
PER_ATTR | other persons attributes (than PER or STATUS) | manspersoon | persons
LOC_NAME    | Name of Location | Sumatra | locations, polities
LOC_ADJ | Derived (adjectival) form of location name | Ternaats | persons, any (through qualification)
ETH_REL | ethno-religious appelation or attribute, not derived from location name | Moor, alfoerees | persons, any (through qualification)
CMTY_NAME   | Name of Commodity | peper | commodities
CMTY_QUAL | Commodity qualifier: colors, processing | gemeen gebleekt | commodities
CMTY_QUANT | Quantity | 840 pikol | commodities
SHIP | Ship name | Abigail | ships
SHIP_TYPE | Ship type | jacht | ships
ORG | Organisation name | Compagnie, Comptoir | organisations, polities
DATE | Date | 14e, ultimo Februari | dates
DOC | Document | Jongste, brief | documents


### PER_NAME
Names of people

- *[Predikant]<sub>PRF</sub> [Petrus Durant]<sub>PER_NAME</sub>*
- *[vaandrig]<sub>PRF</sub> [Francois van den Eijnde]<sub>PER_NAME</sub>*
- *[koningje]<sub>PRF</sub> [Boelo Boelo]<sub>PER_NAME</sub>*
- *[Arou]<sub>PRF</sub> [Teko]<sub>PER_NAME</sub> en [Crain]<sub>PRF</sub> [Jerenica]<sub>PER_NAME</sub>*
- *de [weduwe]<sub>STATUS</sub> van [Tiling]<sub>PER_NAME</sub>*
- *de [Portugesen]<sub>LOC_ADJ</sub> [capiteyn moor]<sub>PRF</sub> [Anthony Hornay]<sub>PER_NAME</sub>*
- *[Mousabeeck]<sub>PER_NAME</sub>, [Ambassadeur]<sub>PRF</sub> den [Conincks]<sub>PRF</sub> van [Persia]<sub>LOC_NAME</sub>*
- *[Antonio van Diemen]<sub>PER_NAME</sub>*

##### Revision June '24
Pay extra attention to long foreign appellations, and how to split these up into PER_NAME/PRF/PER_ATTR, etc.

- *[grave]<sub>PRF</sub> [d’erisseira]<sub>LOC_NAME</sub>, [don]<sub>PER_ATTR</sub> [Louis de menezes]<sub>PER_NAME</sub>*
- *[cap=n de marregerra]<sub>PRF</sub> [don]<sub>PER_ATTR</sub> [Anthonij martinjo de Moura]<sub>PER_NAME</sub>*


### PRF
Professions, titles and functions.
- _[Goegoegoe]<sub>PRF</sub> [Marasaolij Suara Pandjalla]<sub>PER_NAME</sub>_
- _een [Europese]<sub>LOC_ADJ</sub> [Lijfwagt]<sub>PRF</sub>_
- _[princen]<sub>PRF</sub> en [princessen]<sub>PRF</sub> van den bloede_
- _[koning]<sub>PRF</sub> van [Ternaten]<sub>LOC_NAME</sub>_
- _[Scheepsvolck]<sub>PRF</sub>_
- _de [Portugesen]<sub>LOC_ADJ</sub> [capiteyn moor]<sub>PRF</sub> [Anthony Hornay]<sub>PER_NAME</sub>_
- _den [oppercoopman]<sub>PRF</sub> van [Jambij]<sub>LOC_NAME</sub>_
- _[Heer]<sub>PER_ATTR</sub> [Directeur Generaal]<sub>PRF</sub>_
- _[bediendens]<sub>PRF</sub>_

##### Nov. 23 revision
Do not annotate honorific qualifiers such as *Edelen* or *Edelheden*

### STATUS
The status of a person or group of people.
Can be civil or legal status, as well as status of freedom.

Status are attributes which could change but the person might not have control over their change.

- *300 [slauen]<sub>STATUS</sub>*
- *derselver [weduwe]<sub>STATUS</sub> of [Erfgenamen]<sub>STATUS</sub>*
- *alle de [burgers]<sub>STATUS</sub> ende [mardijckers]<sub>ETH_REL</sub>*
- *den [manslaeff]<sub>STATUS</sub> [Pieter van Macasser]<sub>PER_NAME</sub>*
- *een [volwassen]<sub>PER_ATTR</sub> [slaaf]<sub>STATUS</sub>*
- *[wees]<sub>STATUS</sub>*
- *[onderdaan]<sub>STATUS</sub>*

##### Revision Nov. '23
Terms marking status that is linked to a profession should be annotated as professions:
- _[bediendens]<sub>PRF</sub>_

##### Revision Jun. '24
Annotate *juffrouw* as PER_ATTR not as STATUS

Vrund, vriend, and similar terms should not be annotated as an entity. This may be a trigger-word for an event, e.g. [Collaboration](https://github.com/globalise-huygens/nlp-event-detection/wiki#collaboration) or [BeingInARelationship](https://github.com/globalise-huygens/nlp-event-detection/wiki#beinginarelationship). 

### PER_ATTR
Terms used to describe  attributes of people, and are not covered by PER_NAME, PRF, STATUS, ETH_REL, or LOC_ADJ, for instance terms specifying gender. 

<!-- - *[Compes]<sub>ORG</sub> [Volck]<sub>PER_ATTR</sub>* -->
- *een [volwassen]<sub>PER_ATTR</sub> [slaaf]<sub>STATUS</sub>*
- *21 [manspersoonen]<sub>PER_ATTR</sub>, 20 [vrouwen]<sub>PER_ATTR</sub> ende 11 [kinderen]<sub>PER_ATTR</sub>*

Annotate generic titles which indicate PER_ATTR
- *[Intje]<sub>PER_ATTR</sub>*
- *de [Heer]<sub>PER_ATTR</sub>*
- *[cap=n de marregerra]<sub>PRF</sub> [don]<sub>PER_ATTR</sub> [Anthonij martinjo de Moura]<sub>PER_NAME</sub>*

Use PER_ATTR to annotate generic terms which indicate that someone is a person. 
- *[volck]<sub>PER_ATTR</sub>*
- *[koppen]<sub>PER_ATTR</sub>*

Use PER_ATTR to annotate terms of familial relation
- *[moeder]<sub>PER_ATTR</sub>*
- *[zoon]<sub>PER_ATTR</sub>*
- *[grootmoeder]<sub>PER_ATTR</sub>*

##### Revision Nov. '23
Do not annotate personal pronouns, even when they indicate gender, as annotating them systematically will demand special attention (like a round of annotation on coreference).

##### Revision Jun. '24
Annotate generic terms referring to people, e.g. *volck*, *personen*, *koppen*, as PER_ATTR. Annotate generic titles such as *de [heer]<sub>PER_ATTR</sub>*, *[intje]<sub>PER_ATTR</sub>*, or *[don]<sub>PER_ATTR</sub>* 

Annotate *juffrouw* as PER_ATTR not as STATUS

Vrund, vriend, and similar terms should not be annotated as an entity. This may be a trigger-word for an event, e.g. [Collaboration](https://github.com/globalise-huygens/nlp-event-detection/wiki#collaboration) or [BeingInARelationship](https://github.com/globalise-huygens/nlp-event-detection/wiki#beinginarelationship). 

### LOC_NAME
Toponyms, relative locations or descriptors should not be annotated at this point.

- *custe van [Patane]<sub>LOC_NAME</sub>*
- *de Negorij [Goya]<sub>LOC_NAME</sub> aan de overcust van [Halmahera]<sub>LOC_NAME</sub>*
- *het vaerwater van [Mallacca]<sub>LOC_NAME</sub>* 
- *omtrent [poulo percelaer]<sub>LOC_NAME</sub>*

### LOC_ADJ
Derived forms of placenames, these can be adjectives or nouns and can describe people as well as inanimate objects. *This indicates that the term is derived from a placename not necessarily that the person or object is related to that place!*

- *[binase]<sub>LOC_ADJ</sub> Expeditie*
- *de [Jambinesen]<sub>LOC_ADJ</sub>*
- *de [Maccausche]<sub>LOC_ADJ</sub> [Navetten]<sub>SHIP_TYPE</sub>*
- *[Maccause]<sub>LOC_ADJ</sub> [handelaars]<sub>PRF</sub>*
- *een [Europese]<sub>LOC_ADJ</sub> [Lijfwagt]<sub>PRF</sub>*
- *de [Portugesen]<sub>LOC_ADJ</sub> [capiteyn moor]<sub>PRF</sub> [Anthony Hornay]<sub>PER_NAME</sub>*

### ETH_REL
Terms relating to ethnicity (unless this is a location derived term), religion or caste, both in noun or adjectival form.

- *'t [Moorsche]<sub>ETH_REL</sub> [vaartuigh]<sub>SHIP_TYPE</sub>*
- *eenighe [heydensche]<sub>ETH_REL</sub> [vrouwen]<sub>PER_ATTR</sub>*
- *de [Roomsche Christenen]<sub>ETH_REL</sub>*
- *de [Ternataanse]<sub>LOC_ADJ</sub> [alphoeresen]<sub>ETH_REL</sub>*
- *de [Banjaanse]<sub>ETH_REL</sub> [vrouwe]<sub>PER_ATTR</sub>*
- *[Inlander]<sub>ETH_REL</sub>*

##### Revision Nov. '23
Terms marking ethnicity that are derived from locations should be annotated as derived from locations:
- _[kirmanse]<sub>LOC_ADJ</sub>_

##### Revision Jun. '24
Annotate *Inlander*, *inlandsche*, *inboorling*, etc. as ETH_REL

### CMTY_NAME
Commodity without qualifiers or quantities

- *[900 bharen]<sub>CMTY_QUANT</sub> [ppeper]<sub>CMTY_NAME</sub> aldaer tegen [cleeden]<sub>CMTY_NAME</sub> verhandelt*
- *meerder quantitijt [cleeden]<sub>CMTY_NAME</sub>*

Also annotate these when they appear in non-trade contexts

- *[sierie]<sub>PER_NAME</sub> met en [krits]<sub>CMTY_NAME</sub> gewapent*

### CMTY_QUAL
Commodity qualifiers, particularly those describing material, processing, colour.

- *[1200 p:s]<sub>CMTY_QUANT</sub> [roemaals]<sub>CMTY_NAME</sub> [geruijte]<sub>CMTY_QUAL</sub>*
- *[2035 lb]<sub>CMTY_QUANT</sub> [koper]<sub>CMTY_NAME</sub> [in Plaaten]<sub>CMTY_QUAL</sub>*
- *[24 Paar]<sub>CMTY_QUANT</sub> [zijde]<sub>CMTY_QUAL</sub> [koussen]<sub>CMTY_NAME</sub>*

Separate out multiple qualifiers
- *[800 „]<sub>CMTY_QUANT</sub> [Cassa]<sub>CMTY_NAME</sub> [fijne]<sub>CMTY_QUAL</sub> [met Goude hoofden]<sub>CMTY_QUAL</sub>*

Commodity qualifiers serve to identify commodities more precisely. Do NOT annotate circumstantial qualifiers like *geeyste*: 
- *geeyste [houtwercken]<sub>CMTY_NAME</sub>*


### CMTY_QUANT
Used to annotate quantities as numerical value and their related units. Units include currencies as well as terms such as *stuks* or *paar*. Prices are annotated as CMTY_QUANT


- *[20 ps]<sub>CMTY_QUANT</sub> [roggevellen]<sub>CMTY_NAME</sub>*
- *[ƒ 120885„15„2]<sub>CMTY_QUANT</sub>*
- *[32000 pagoden]<sub>CMTY_QUANT</sub>*
- *[10 a 12 packen]<sub>CMTY_QUANT</sub>*
- *[450 „]<sub>CMTY_QUANT</sub> [Salempoeris]<sub>CMTY_NAME</sub> [blaauwe]<sub>CMTY_QUAL</sub>*

Quantities should be precise. Do NOT annotate vague quantities like *goede partije*:
- *een goede partije [peper]<sub>CMTY_NAME</sub>*

##### Nov. 23 revision: 
* note that implicit units marked by '„' are to be included too:
  * *[450 „]<sub>CMTY_QUANT</sub> [Salempoeris]<sub>CMTY_NAME</sub> [blaauwe]<sub>CMTY_QUAL</sub>*
* also include dots at the end of monetary values:
  * *[ƒ 38830:12: 8.]<sub>CMTY_QUANT</sub>*


### SHIP
Used to annotate ship names.

- *de [fluijt]<sub>SHIP_TYPE</sub> [Amsterveen]<sub>SHIP</sub>*
- *met [Jachten]<sub>SHIP_TYPE</sub> [Grootenbrouck]<sub>SHIP</sub>, [Brouwershaven]<sub>SHIP</sub> ende [Battavia]<sub>SHIP</sub>*
- *de [Leeuwinne]<sub>SHIP</sub> ende [Kemphaen]<sub>SHIP</sub> waren aldaer noch niet verschenen*
- *twee [Engelse]<sub>LOC_ADJ</sub> [schepen]<sub>SHIP_TYPE</sub>, het eene genaemt de [Luypaerd]<sub>SHIP</sub>, het ander genaemt de [Madras]<sub>SHIP</sub>*
- *de [Hollandse Tuin]<sub>SHIP</sub>*
- *de [chialoupen]<sub>SHIP_TYPE</sub> de [Javaensen Houtcoper]<sub>SHIP</sub> en de [Diamant]<sub>SHIP</sub>*

### SHIP_TYPE
Used to annotate ship types. This includes generic terms such as "schip".

- *de [Maccausche]<sub>LOC_ADJ</sub> [Navetten]<sub>SHIP_TYPE</sub>*
- *[chialoep]<sub>SHIP_TYPE</sub> de [doradus]<sub>SHIP</sub>*
- *[hoeker]<sub>SHIP_TYPE</sub> de [nijptang]<sub>SHIP</sub>*

### ORG
Used to annotate organizations. This includes references to the VOC or other East India Companies.

- *[Raad van Justitie]<sub>ORG</sub>*
- *[Heren Seventhiene]<sub>ORG</sub>*
- *dit [comptoir]<sub>ORG</sub>*

If the name contains a placename annotate the placename separately. e.g.
- *de [weescamer]<sub>ORG</sub> van [Batavia]<sub>LOC_NAME</sub>*
- *een [goerab]<sub>SHIP_TYPE</sub> van d' [Engelse]<sub>LOC_ADJ</sub> [Comps]<sub>ORG</sub>*
- *een [deens]<sub>LOC_ADJ</sub> [comp„s]<sub>ORG</sub> [scheepje]<sub>SHIP_TYPE</sub>*

### DATE
Used to annotate dates. 

- *[a:o 1628]<sub>DATE</sub>*
- *[11: september passado]<sub>DATE</sub>*
- *[20en Nouembr]<sub>DATE</sub>*
- *[den 25en ditto]<sub>DATE</sub>*
- *[8 januarij deses jaars]<sub>DATE</sub>*
- *[P:mo 9ber]<sub>DATE</sub>*
- *tegen [Martij toecomende]<sub>DATE</sub>*
- *[26en ditto]<sub>DATE</sub>*

Split up coordinated elements into separate annotations
- *den [15en]<sub>DATE</sub> en [17en december]<sub>DATE</sub>*

Do NOT annotate generic or vague time expressions
- *ten spoedigste*

##### Nov. '23 revision
Annotate relative dates and seasons:
- _[Eergisteren]<sub>DATE</sub>_
- _[a=o pass„o]<sub>DATE</sub>_
- _[mousson]<sub>DATE</sub>_

### DOC
Used to annotate documents. Annotate generic terms in as much as possible

- *de successive [contracten]<sub>DOC</sub> vernieuwt*
- *als reets bij de [brieven]<sub>DOC</sub> der vorige jaren vermelt zijnde*
- *[generale brieven]<sub>DOC</sub>, van [30„en septemb.]<sub>DATE</sub> en [4en November des verleden]<sub>DATE</sub> mitsgaders [8„en Maert deses Jaers]<sub>DATE</sub>*
- *met onse [Jongste]<sub>DOC</sub> is UEd geaduiseert*

Annotate generic terms in as much as possible, without specific qualifiers: 
- *d„o [missive]<sub>DOC</sub> in Copia*
