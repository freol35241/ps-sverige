"""Manually distilled position statements per (topic, party).

Each statement is a faithful ~40–60 word distillation of one or more actual
reservations (or 'Utskottets ställningstagande' for cabinet parties), written
to be:
  - self-contained: a voter can evaluate it without context
  - distinctive: differs in content, not just in writing style
  - clean Swedish: no procedural boilerplate, no yrkande citations
  - source-attributed: the dok_id of the underlying betänkande is recorded

The distillations are written by hand (not LLM-generated). The export
pipeline (scripts/export_for_web.py) prefers these over the algorithmic
excerpts where available.

Voice convention:
  - Opposition (V, S, MP, C, SD): first person plural "vi" or "vårt parti"
  - Cabinet (L, KD, M): all three get the same distillations since they
    vote and reserve as one — they share majority-opinion content.
"""

# Format:
#   CURATED[topic_id][party] = [
#       {"text": "...", "source": "DOK_ID1, DOK_ID2"},
#       ...
#   ]


CURATED: dict[int, dict[str, list[dict]]] = {}


# ============================================================================
# Topic 13 — Migration och asylpolitik
# ============================================================================
CURATED[13] = {
    "V": [
        {
            "text": "Den tillfälliga utlänningslagen har inte gjort asylsystemet "
                    "mer hållbart eller mänskligt — tvärtom. Vi vill återgå till "
                    "utlänningslagen som den såg ut innan, så att asylrätten "
                    "och de konventioner Sverige har bundit sig vid återigen "
                    "respekteras. Det är en rättighetsfråga, inte en "
                    "förhandlingsfråga.",
            "source": "HC01SfU18",
        },
        {
            "text": "Ensamkommande barn och unga som varit i Sverige i över ett år "
                    "ska beviljas permanent uppehållstillstånd. För personer som "
                    "har levt här länge utan tillstånd bör någon form av "
                    "regularisering utredas. Människor ska inte tvingas leva "
                    "decennier i ovisshet på grund av rättsosäkra processer.",
            "source": "HD01SfU16",
        },
    ],
    "S": [
        {
            "text": "Det bör ställas tydliga kunskapskrav i språk och "
                    "samhällskunskap för permanent uppehållstillstånd. Samtidigt "
                    "är det inte rimligt att jämställa gängkriminella med "
                    "brottsoffer som utnyttjats för exempelvis sexuella ändamål "
                    "när vi prövar utvisning på grund av bristande vandel. "
                    "Begreppet behöver moderniseras.",
            "source": "HA01SfU15",
        },
        {
            "text": "Arbetet med återvändande av personer med avlägsnandebeslut "
                    "behöver intensifieras. Den förra regeringen införde "
                    "konkreta målsättningar för återvändandet — sedan dess har "
                    "myndigheterna saknat tydlig riktning. Vi vill återinföra "
                    "mål och stärka samarbetena med tredjeländer för att få "
                    "fler beslut att verkställas.",
            "source": "HC01SfU18",
        },
    ],
    "MP": [
        {
            "text": "Asylrätten är grundläggande och måste värnas. Barn ska inte "
                    "tas i förvar — det är inhumant. Sverige bör införa lagliga "
                    "vägar för att söka asyl, exempelvis genom viseringar för "
                    "personer som har för avsikt att ansöka i EU. "
                    "Barnkonventionen måste få fullt genomslag i utlänningslagen.",
            "source": "HA01SfU15, HD01SfU16",
        },
        {
            "text": "Personer som arbetat för utländska styrkor i Afghanistan "
                    "förföljs nu av talibanerna. Cirka 50 personer som uppfyller "
                    "kraven blev inte vidarebosatta — regeringen bör snarast "
                    "fatta de beslut som krävs för att ge dem och deras familjer "
                    "skydd i Sverige.",
            "source": "HC01SfU18",
        },
    ],
    "C": [
        {
            "text": "Regleringen av arbetskraftsinvandring bör utgå från "
                    "företagens faktiska kompetensbehov. Arbetslivskriminalitet "
                    "ska bekämpas genom effektiva riktade kontroller — inte "
                    "genom generella lönegolv som stänger ute hela yrkesgrupper. "
                    "Arbetskraftsinvandringen är värdefull för Sveriges ekonomi.",
            "source": "HD01SfU12",
        },
        {
            "text": "Personer som söker uppehållstillstånd på grund av "
                    "anknytning ska informeras om referenspersonens tidigare "
                    "våldsbrott i de fall det finns en risk för att de utnyttjas. "
                    "Det handlar om att skydda människor — främst kvinnor — som "
                    "kommer till Sverige i utsatt position.",
            "source": "HA01SfU15",
        },
    ],
    "L": [
        {
            "text": "För att återupprätta förtroendet för den reglerade "
                    "invandringen krävs en ansvarsfull, stram och långsiktigt "
                    "hållbar migrationspolitik. Samtidigt ska asylrätten "
                    "upprätthållas. Sverige bör arbeta för ett europeiskt "
                    "asylsystem med stark gränskontroll och fungerande "
                    "screeningförfarande.",
            "source": "HA01SfU15",
        },
        {
            "text": "Det svenska medborgarskapet ska tillmätas stor betydelse, "
                    "såväl rättsligt som symboliskt. Kraven bör skärpas för "
                    "förvärv av medborgarskap genom anmälan så att de omfattar "
                    "grundläggande krav på skötsamhet och respekt för "
                    "samhället.",
            "source": "HB01SfU15",
        },
    ],
    "KD": [
        # Cabinet shares the same distillations — they vote identically and the
        # majority opinion is collective. We register them per party so the
        # export still has cells filled, but the content is the same.
        {
            "text": "För att återupprätta förtroendet för den reglerade "
                    "invandringen krävs en ansvarsfull, stram och långsiktigt "
                    "hållbar migrationspolitik. Samtidigt ska asylrätten "
                    "upprätthållas. Sverige bör arbeta för ett europeiskt "
                    "asylsystem med stark gränskontroll och fungerande "
                    "screeningförfarande.",
            "source": "HA01SfU15",
        },
        {
            "text": "Sverige har under senare år haft en omfattande "
                    "arbetskraftsinvandring till lågkvalificerade yrken — ofta "
                    "till arbeten som kan utföras av personer som redan bor "
                    "här. Missbruket av regelverket är omfattande. En "
                    "omställning med skärpta regler är nödvändig.",
            "source": "HA01SfU17, HB01SfU11",
        },
    ],
    "M": [
        {
            "text": "Sverige har under senare år haft en omfattande "
                    "arbetskraftsinvandring till lågkvalificerade yrken — ofta "
                    "till arbeten som kan utföras av personer som redan bor "
                    "här. Missbruket av regelverket är omfattande. En "
                    "omställning med skärpta regler är nödvändig.",
            "source": "HA01SfU17, HB01SfU11",
        },
        {
            "text": "För att återupprätta förtroendet för den reglerade "
                    "invandringen krävs en ansvarsfull, stram och långsiktigt "
                    "hållbar migrationspolitik. Samtidigt ska asylrätten "
                    "upprätthållas. Sverige bör arbeta för ett europeiskt "
                    "asylsystem med stark gränskontroll och fungerande "
                    "screeningförfarande.",
            "source": "HA01SfU15",
        },
    ],
    "SD": [
        {
            "text": "Människor på flykt ska få skydd — men de ska söka asyl i "
                    "det första säkra land de kan nå. Sverige ska i "
                    "normalfallet ta emot asylsökande enbart i en situation där "
                    "ett krig eller en djupare kris har brutit ut i något av "
                    "våra grannländer eller i vårt omedelbara närområde, som i "
                    "Ukraina.",
            "source": "HC01SfU18, HA01SfU15",
        },
        {
            "text": "Anhöriginvandring till särskilt utsatta områden bör "
                    "förbjudas. Allt fler stadsdelar präglas av utanförskap, "
                    "segregation, hög arbetslöshet och kriminalitet — det är "
                    "inte rimligt att fortsätta öka belastningen på de mest "
                    "ansträngda områdena.",
            "source": "HA01SfU15",
        },
    ],
}


# ============================================================================
# Topic 18 — Klimat och energi
# ============================================================================
CURATED[18] = {
    "V": [
        {
            "text": "Klimatkrisen är akut och kräver ett systemskifte. Stora "
                    "delar av samhället måste elektrifieras, vilket innebär den "
                    "största utbyggnaden av elproduktion och eldistribution i "
                    "modern tid. Det kräver långsiktiga, välgrundade beslut — "
                    "och samtidigt energieffektivisering, eftersom Sverige inte "
                    "kan bygga sig ur problemet med bara mer el.",
            "source": "HD01NU13",
        },
        {
            "text": "Reduktionsplikten är ett av de viktigaste verktygen för "
                    "att minska transportsektorns utsläpp. En höjning från 6 "
                    "till 10 procent är otillräcklig för att klara EU:s krav "
                    "och de nationella klimatmålen. Ökad inhemsk produktion av "
                    "förnybara drivmedel stärker dessutom Sveriges "
                    "självförsörjning.",
            "source": "HC01MJU17",
        },
    ],
    "S": [
        {
            "text": "Klimatklivet — som vi införde 2015 — har med över 10 "
                    "miljarder kronor stöttat 4 500 lokala klimatåtgärder och "
                    "minskat utsläppen med 2,4 miljoner ton om året. Stödet "
                    "ska fortsätta utvecklas, men med en tydligare inriktning "
                    "på att underlätta klimatomställning för vanligt folk.",
            "source": "HA01MJU16",
        },
        {
            "text": "Energisystemet är centralt för Sveriges utveckling. En "
                    "framgångsrik elektrifiering är avgörande för att nå "
                    "nettonollutsläpp 2045 och samtidigt skapa tillväxt. Billig "
                    "el är en konkurrensfördel för svenska företag — och en "
                    "förutsättning för att hushållens ekonomi inte ska gröpas "
                    "ur.",
            "source": "HD01NU13",
        },
    ],
    "MP": [
        {
            "text": "Vår grundläggande syn är att samhället måste vara "
                    "resurseffektivt och bygga på ansvarsfullt tillvaratagande "
                    "av jordens tillgångar. Cirkulär ekonomi är ett centralt "
                    "verktyg. Att fortsätta bränna fossila energitillgångar "
                    "går definitivt inte ihop med detta tankesätt — vi behöver "
                    "stora satsningar på förnybar produktion.",
            "source": "HD01NU13",
        },
        {
            "text": "Reduktionsplikten är en nödvändig övergångslösning medan "
                    "fordonsflottan ställs om till el. Den ger en kraftig och "
                    "förutsägbar sänkning av klimatpåverkan från befintliga "
                    "fordon. Den föreslagna nivån på 10 procent är "
                    "otillräcklig — flera tunga remissinstanser har varnat för "
                    "att klimatmålen inte nås.",
            "source": "HC01MJU17",
        },
    ],
    "C": [
        {
            "text": "Klimatomställningen är vår tids största globala utmaning. "
                    "Beroendet av fossil energi måste brytas en gång för alla — "
                    "inte minst eftersom importen ofta sker från diktaturer. "
                    "Hoten mot klimatet ska mötas med snabb elektrifiering "
                    "baserad på gröna, hållbara energilösningar.",
            "source": "HD01NU13",
        },
        {
            "text": "Klimatstöd som Klimatklivet och Industriklivet behöver "
                    "vassa kontrollmekanismer så att skattepengar går till "
                    "klimateffektiva projekt. Teknikneutralitet ska vägleda, "
                    "och satsningarna bör breddas till exempelvis vätgas och "
                    "CCU — tekniker som möjliggör återanvändning av infångad "
                    "koldioxid.",
            "source": "HA01MJU16",
        },
    ],
    "L": [
        {
            "text": "Sveriges konkurrenskraft och välfärd bygger på tillgång "
                    "till fossilfri energi till konkurrenskraftiga priser. "
                    "Omställningen till fossilfri energi är central för "
                    "klimatomställningen — och kräver en bred energimix där "
                    "kärnkraften har en nyckelroll.",
            "source": "HB01MJU15",
        },
        {
            "text": "Elektrifieringen av fordonsflottan är av stor betydelse "
                    "för att minska utsläppen. Vägtransporter står för en "
                    "femtedel av världens växthusgasutsläpp, och flera stora "
                    "tillverkare har långtgående planer på nollutsläppsfordon. "
                    "Sverige bör utnyttja den teknikutvecklingen.",
            "source": "HB01MJU15",
        },
    ],
    "KD": [
        {
            "text": "Sveriges konkurrenskraft och välfärd bygger på tillgång "
                    "till fossilfri energi till konkurrenskraftiga priser. "
                    "Omställningen till fossilfri energi är central för "
                    "klimatomställningen — och kräver en bred energimix där "
                    "kärnkraften har en nyckelroll.",
            "source": "HB01MJU15",
        },
        {
            "text": "Den gröna omställningen innebär en strukturomvandling för "
                    "stora delar av svenskt näringsliv. Sverige måste fortsätta "
                    "vara en ledande kunskapsnation. Vi behöver fler människor "
                    "med gedigna kunskaper inom naturvetenskap, teknik och "
                    "matematik — och en STEM-strategi som spänner över hela "
                    "utbildningssystemet.",
            "source": "HB01MJU15",
        },
    ],
    "M": [
        {
            "text": "Sveriges konkurrenskraft och välfärd bygger på tillgång "
                    "till fossilfri energi till konkurrenskraftiga priser. "
                    "Omställningen till fossilfri energi är central för "
                    "klimatomställningen — och kräver en bred energimix där "
                    "kärnkraften har en nyckelroll.",
            "source": "HB01MJU15",
        },
        {
            "text": "Elektrifieringen av fordonsflottan är av stor betydelse "
                    "för att minska utsläppen. Vägtransporter står för en "
                    "femtedel av världens växthusgasutsläpp, och flera stora "
                    "tillverkare har långtgående planer på nollutsläppsfordon. "
                    "Sverige bör utnyttja den teknikutvecklingen.",
            "source": "HB01MJU15",
        },
    ],
    "SD": [
        {
            "text": "Klimatmål ska formuleras i termer av det miljöproblem vi "
                    "vill lösa, inte i termer av de medel vi väljer. Därför bör "
                    "alla sektorsvisa mål inom klimatpolitiken avvecklas. Mål "
                    "om andel förnybar energi och koldioxidfri fordonsflotta "
                    "bör överges. Politiken ska syfta till kostnadseffektiva "
                    "globala utsläppsminskningar.",
            "source": "HA01MJU16",
        },
        {
            "text": "Vattenkraften är central för Sveriges fossilfria "
                    "elförsörjning, både som produktionskälla och som "
                    "reglerkraft. Dess unika förmåga att balansera elsystemet "
                    "har dock börjat slå i taket när den väderberoende "
                    "kraftproduktionen växer. Pumpkraft i befintliga magasin "
                    "har stor potential.",
            "source": "HC01NU19, HB01NU14",
        },
    ],
}


# ============================================================================
# Topic 7 — Skola och lärare
# ============================================================================
CURATED[7] = {
    "V": [
        {
            "text": "Demokratiskt fattade beslut och behovsbedömning ska vara "
                    "grunden för nyetablering av skolor — så att skattemedel "
                    "inte slösas bort på överetablering. Skolkoncerner kan i "
                    "dag enkelt föra pengar till andra bolag i koncernen eller "
                    "ut till närstående bolag genom att fakturera för diverse "
                    "tjänster. Skattepengar ska gå till elevernas utbildning.",
            "source": "HC01UbU9",
        },
        {
            "text": "Vissa privata skolkoncerner har undantag från kravet på "
                    "lärarlegitimation eftersom de undervisar på engelska. "
                    "Skolinspektionen har visat att bristen på behörig "
                    "personal gör att skolorna misslyckas med läroplanens "
                    "krav på kvalitet. Det måste förändras — undantagen bör "
                    "tas bort.",
            "source": "HB01UbU11",
        },
    ],
    "S": [
        {
            "text": "Den kraftiga expansionen av vinstdrivande skolor har "
                    "lett till ökad vinstjakt och minskad demokratisk kontroll "
                    "över skolan. Skolmarknaden domineras av stora koncerner "
                    "med starka juridiska resurser. När barnkullarna nu krymper "
                    "förvärras problemen — staten måste dra i nödbromsen.",
            "source": "HC01UbU9",
        },
        {
            "text": "Att skolplikten upphör efter grundskolan ger unga en "
                    "falsk signal om att de inte behöver utbilda sig. Med dagens "
                    "krav på språk, matematik och digitala kompetenser finns "
                    "ingen arbetsmarknad för unga som inte läser gymnasiet. "
                    "Hela samhället har ansvar att inte lämna någon utan vettig "
                    "sysselsättning.",
            "source": "HD01UbU7",
        },
    ],
    "MP": [
        {
            "text": "Det är hög tid att förbjuda vinstutdelande aktiebolagsskolor "
                    "och få bort marknadslogiken ur skolan. Till dess att ett "
                    "vinststopp har införts behövs ett omedelbart "
                    "etableringsstopp för nya vinstdrivande friskolor. "
                    "Befintligt utbud ska väga tyngre vid beslut om "
                    "etablering.",
            "source": "HC01UbU9",
        },
        {
            "text": "Staten bör ta ett större ansvar för finansieringen av "
                    "skolan och förskolan. Det befintliga statliga "
                    "likvärdighetsbidraget för grundskolan behöver förstärkas, "
                    "och även förskolans likvärdighet behöver tryggas genom "
                    "ökad statlig andel av finansieringen. Riktade bidrag bör "
                    "bli färre och enklare.",
            "source": "HC01UbU9",
        },
    ],
    "C": [
        {
            "text": "Skolplikten bör reformeras för att möjliggöra obligatoriska "
                    "stödinsatser och anpassningar. Insatserna ska bygga på "
                    "evidens och utgå från regelbundna kartläggningar av "
                    "elevernas kunskaper. När mindre insatser inte räcker ska "
                    "mer omfattande sättas in — som intensivundervisning under "
                    "helger och lov.",
            "source": "HD01UbU7",
        },
        {
            "text": "Det är lärarna som gör skillnad för elevens "
                    "kunskapsinhämtning. Tre av tio lärare i grundskolan saknar "
                    "behörighet. Vi vill ställa krav på huvudmännen att ta fram "
                    "en utbildningsplan för varje anställd obehörig lärare. Den "
                    "administrativa kontrollen av rektorerna behöver samtidigt "
                    "minska.",
            "source": "HD01UbU9, HB01UbU9",
        },
    ],
    "L": [
        {
            "text": "Det förtydligas i skollagen att eleverna ska ha tillgång "
                    "till läroböcker, andra läromedel och lärverktyg som "
                    "behövs för en god kunskapsutveckling. Läromedlens "
                    "främsta funktion är att stärka elevers kunskapsutveckling "
                    "och förhålla sig till kurs- och ämnesplaner.",
            "source": "HB01UbU6",
        },
        {
            "text": "Skolinspektionens uppdrag är att se till att skolhuvudmän — "
                    "kommunala eller fristående — sköter sin verksamhet enligt "
                    "lag. Inspektionen ställer krav och bidrar till att "
                    "huvudmännen följer skollagen. Det är denna ordnings "
                    "logik vi avvaktar och stärker, snarare än att lägga om "
                    "huvudmannaskapet.",
            "source": "HD01UbU7",
        },
    ],
    "KD": [
        {
            "text": "Det förtydligas i skollagen att eleverna ska ha tillgång "
                    "till läroböcker, andra läromedel och lärverktyg som "
                    "behövs för en god kunskapsutveckling. Läromedlens "
                    "främsta funktion är att stärka elevers kunskapsutveckling "
                    "och förhålla sig till kurs- och ämnesplaner.",
            "source": "HB01UbU6",
        },
        {
            "text": "Skolinspektionens uppdrag är att se till att skolhuvudmän — "
                    "kommunala eller fristående — sköter sin verksamhet enligt "
                    "lag. Inspektionen ställer krav och bidrar till att "
                    "huvudmännen följer skollagen. Det är denna ordnings "
                    "logik vi avvaktar och stärker, snarare än att lägga om "
                    "huvudmannaskapet.",
            "source": "HD01UbU7",
        },
    ],
    "M": [
        {
            "text": "Det förtydligas i skollagen att eleverna ska ha tillgång "
                    "till läroböcker, andra läromedel och lärverktyg som "
                    "behövs för en god kunskapsutveckling. Läromedlens "
                    "främsta funktion är att stärka elevers kunskapsutveckling "
                    "och förhålla sig till kurs- och ämnesplaner.",
            "source": "HB01UbU6",
        },
        {
            "text": "Skolinspektionens uppdrag är att se till att skolhuvudmän — "
                    "kommunala eller fristående — sköter sin verksamhet enligt "
                    "lag. Inspektionen ställer krav och bidrar till att "
                    "huvudmännen följer skollagen. Det är denna ordnings "
                    "logik vi avvaktar och stärker, snarare än att lägga om "
                    "huvudmannaskapet.",
            "source": "HD01UbU7",
        },
    ],
    "SD": [
        {
            "text": "Nationella riktlinjer ska fastslås så att digitala verktyg "
                    "tas bort helt för årskurs F–3. I årskurs 4–6 ska "
                    "datorsalar användas i stället för personliga paddor. När "
                    "elever får personliga datorer i högstadiet ska de vara "
                    "komplement, inte ersättare till fysiska läroböcker.",
            "source": "HB01UbU9",
        },
        {
            "text": "Mindre skolor utanför tätorterna bör så långt det är "
                    "möjligt bevaras eftersom det möjliggör för familjer att "
                    "bo där. Det är lika genomförbart att skjutsa barn från "
                    "tätorternas ytterområden till en landsbygdsskola som "
                    "tvärtom. Ett statligt bryggstöd bör införas vid "
                    "tillfälligt vikande elevunderlag.",
            "source": "HC01UbU9",
        },
    ],
}


# ============================================================================
# Topic 3 — Brott och kriminalvård
# ============================================================================
CURATED[3] = {
    "V": [
        {
            "text": "Polisens brottsförebyggande arbete är eftersatt — både i "
                    "metoder och resurser — trots att det är ett centralt "
                    "område. Bristerna gäller beprövade metoder, dokumentation "
                    "och utvärdering. Polisens trygghets- och kontaktskapande "
                    "arbete behöver ett kraftigt resurstillskott och en "
                    "tydligare balans mellan utryckning, utredning och "
                    "förebyggande.",
            "source": "HC01JuU15",
        },
        {
            "text": "Hela Kriminalvårdens organisation behöver ses över. "
                    "Huvudkontoret har fått växa obehindrat på bekostnad av "
                    "anstalterna; arbetsmiljön har stora brister; fackförbund "
                    "har larmat om tystnadskultur. Det handlar inte bara om "
                    "platsbrist — det handlar om en verksamhet under stark "
                    "press.",
            "source": "HB01JuU17",
        },
    ],
    "S": [
        {
            "text": "Tekniska framsteg gör det möjligt att klara upp brott även "
                    "efter lång tid — preskriptionstiderna behöver därför "
                    "ändras. Preskriptionsutredningen har lagt förslag som "
                    "behöver behandlas. Vi vill att regeringen skyndsamt "
                    "återkommer till riksdagen med lagförslag om förändrade "
                    "preskriptionsregler.",
            "source": "HA01JuU11",
        },
        {
            "text": "Återfallsförebyggande behandling är central för att bryta "
                    "kriminalitet. Riksrevisionen visar att en låg andel av de "
                    "intagna som behöver det får tillgång till program — och "
                    "att överbeläggningar gör det svårt att upprätthålla en "
                    "stödjande miljö. Platsbristen kommer att förvärras och "
                    "det krävs handling.",
            "source": "HC01JuU19",
        },
    ],
    "MP": [
        {
            "text": "Beslut enligt inhämtningslagen bör hanteras av domstol — "
                    "inte av Åklagarmyndigheten, som inte uppfyller det krav "
                    "på oberoende som EU-domstolen ställer upp. Både JO och "
                    "JK har efterfrågat en djupare analys av förslagets "
                    "förenlighet med unionsrätten. Rättssäkerheten kring "
                    "tvångsmedel måste vara orubblig.",
            "source": "HB01JuU24",
        },
        {
            "text": "Situationen på fängelserna är allvarlig. Utan adekvat "
                    "rehabilitering riskerar en fängelsevistelse att leda till "
                    "fortsatt marginalisering och kriminalitet. Kriminalvårdens "
                    "uppdrag att erbjuda behandling är inte bara önskvärt — "
                    "det är centralt för att minska återfallen i brott.",
            "source": "HC01JuU19",
        },
    ],
    "C": [
        {
            "text": "Tekniska framsteg och en allmänt ändrad syn på preskription "
                    "påkallar förändring av preskriptionstiderna. "
                    "Preskriptionstiden bör slopas inte bara för "
                    "sexualbrott mot barn, utan även för sexuellt utnyttjande "
                    "av barn och grovt sexuellt övergrepp mot barn. "
                    "Preskriptionsutredningens förslag bör beredas skyndsamt.",
            "source": "HA01JuU11",
        },
        {
            "text": "Sverige saknar nationell beredskap när personuppgifter "
                    "läcker. Tusentals människor lever med skyddad identitet — "
                    "kvinnor som flytt våld, vittnen, hotade människor. För "
                    "dem är skyddet bokstavligen en fråga om liv eller död. "
                    "Vid de stora läckorna det senaste året har stödet varit "
                    "närmast obefintligt.",
            "source": "HD01JuU12",
        },
    ],
    "L": [
        {
            "text": "Våld och förtryck har ingen plats i vårt samhälle. Mäns "
                    "våld mot kvinnor och våld i nära relationer ska "
                    "förebyggas, motverkas och bekämpas på bred front. Våldet "
                    "kan ta sig många uttryck — fysiskt, psykiskt, ekonomiskt, "
                    "sexuellt — och alla ska ha samma rätt att bestämma över "
                    "sitt liv, sin kropp och sin partner.",
            "source": "HC01AU9",
        },
        {
            "text": "Det är upp till Polismyndigheten att besluta om vilka "
                    "fordon polisen ska använda, hur uniformerna ska se ut och "
                    "hur vapenanvändningen ska regleras. Det befintliga "
                    "regelverket för polisens nödvärnsrätt är ändamålsenligt "
                    "utformat — det ska inte göras om utifrån enstaka motioner.",
            "source": "HA01JuU10",
        },
    ],
    "KD": [
        {
            "text": "Våld och förtryck har ingen plats i vårt samhälle. Mäns "
                    "våld mot kvinnor och våld i nära relationer ska "
                    "förebyggas, motverkas och bekämpas på bred front. Våldet "
                    "kan ta sig många uttryck — fysiskt, psykiskt, ekonomiskt, "
                    "sexuellt — och alla ska ha samma rätt att bestämma över "
                    "sitt liv, sin kropp och sin partner.",
            "source": "HC01AU9",
        },
        {
            "text": "Det är upp till Polismyndigheten att besluta om vilka "
                    "fordon polisen ska använda, hur uniformerna ska se ut och "
                    "hur vapenanvändningen ska regleras. Det befintliga "
                    "regelverket för polisens nödvärnsrätt är ändamålsenligt "
                    "utformat — det ska inte göras om utifrån enstaka motioner.",
            "source": "HA01JuU10",
        },
    ],
    "M": [
        {
            "text": "Våld och förtryck har ingen plats i vårt samhälle. Mäns "
                    "våld mot kvinnor och våld i nära relationer ska "
                    "förebyggas, motverkas och bekämpas på bred front. Våldet "
                    "kan ta sig många uttryck — fysiskt, psykiskt, ekonomiskt, "
                    "sexuellt — och alla ska ha samma rätt att bestämma över "
                    "sitt liv, sin kropp och sin partner.",
            "source": "HC01AU9",
        },
        {
            "text": "Det är upp till Polismyndigheten att besluta om vilka "
                    "fordon polisen ska använda, hur uniformerna ska se ut och "
                    "hur vapenanvändningen ska regleras. Det befintliga "
                    "regelverket för polisens nödvärnsrätt är ändamålsenligt "
                    "utformat — det ska inte göras om utifrån enstaka motioner.",
            "source": "HA01JuU10",
        },
    ],
    "SD": [
        {
            "text": "Livstids fängelse är den allvarligaste påföljd vi har, "
                    "men det förekommer att livstidsstraff omvandlas till "
                    "tidsbestämda. Det innebär att påföljden blir betydligt "
                    "skonsammare än ett verkligt livstidsstraff. Vi vill "
                    "införa livstids fängelse utan möjlighet till omvandling — "
                    "för brott som i dag kan ge livstid.",
            "source": "HA01JuU11",
        },
        {
            "text": "Rörelseinskränkande föreskrifter ska kunna kombineras med "
                    "helghemarrest för ungdomar som dömts till "
                    "ungdomsövervakning. Dagens lagstiftning täcker inte det "
                    "behov av övervakning och kontroll som kan finnas — "
                    "särskilt på helgdagar då riskmiljöer förekommer. "
                    "Lagstiftningen behöver revideras.",
            "source": "HA01JuU15",
        },
    ],
}


# ============================================================================
# Topic 6 — Skatter
# ============================================================================
CURATED[6] = {
    "V": [
        {
            "text": "Drivmedelsskatt är ett viktigt klimatpolitiskt verktyg men "
                    "trubbigt — det skiljer inte mellan en höginkomsttagare i "
                    "staden och en låginkomsttagare i glesbygden utan "
                    "kollektivtrafikalternativ. Styrmedel måste utformas så att "
                    "klyftorna mellan stad och landsbygd inte växer, och "
                    "kompensatoriska åtgärder behövs när skatter ger negativa "
                    "fördelningseffekter.",
            "source": "HB01SkU13",
        },
        {
            "text": "Försvarsupprustningen ska bäras av dem som har bäst "
                    "ekonomiska förutsättningar. Vi vill därför utreda en ny "
                    "beredskapsskatt. Det krisläge som Putins krig i Ukraina "
                    "skapar kräver ökade försvarsanslag — men finansieringen "
                    "ska inte tas från välfärden.",
            "source": "HA01SkU12",
        },
    ],
    "S": [
        {
            "text": "Skattesystemets huvudsakliga uppgift är att finansiera "
                    "välfärden. Reglerna ska vara generella, med breda "
                    "skattebaser och väl avvägda skattesatser. Utöver det ska "
                    "skattesystemet bidra till omfördelning, jämlikhet och "
                    "jämställdhet. Tydliga regler stärker legitimiteten och "
                    "minskar utrymmet för fusk.",
            "source": "HB01SkU11",
        },
        {
            "text": "Organiserade momsbedrägerier — karusellhandeln — kostar "
                    "staten miljarder varje år och hotar både skattesystemets "
                    "legitimitet och seriösa företags konkurrensvillkor. "
                    "Regeringen har sedan maj 2024 haft en färdig utredning "
                    "med konkreta förslag. Det är dags att gå vidare med "
                    "åtgärderna.",
            "source": "HD01SkU17",
        },
    ],
    "MP": [
        {
            "text": "Mensskydd är inte ett val — det är en biologisk realitet. "
                    "EU:s medlemsstater fattade 2016 beslut om att momsen på "
                    "tamponger och bindor kan sänkas till noll. Vi vill att "
                    "Sverige tar bort den momsen. För en kvinna som "
                    "menstruerar i 40 år kan kostnaden uppgå till 48 000 "
                    "kronor.",
            "source": "HA01SkU15",
        },
        {
            "text": "Skatten på egenproducerad el och andelsägande av "
                    "solenergi bör sänkas — och slopas helt för hushåll. "
                    "Vindkraften är redan Sveriges billigaste energikälla; "
                    "solcellerna på villatak gör hushållen mindre beroende. "
                    "Skattesystemet ska underlätta — inte hämma — den gröna "
                    "omställningen.",
            "source": "HA01SkU14, HA01SkU11",
        },
    ],
    "C": [
        {
            "text": "Ju fler människor som jobbar, desto bättre fungerar den "
                    "offentliga ekonomin. Sänkt skatt skapar bättre "
                    "förutsättningar för jobb och företagande — och därmed "
                    "långsiktigt mer resurser till välfärden. Sänkta "
                    "arbetsgivaravgifter är en viktig del för att skapa "
                    "utrymme för företag att anställa fler.",
            "source": "HA01SkU13",
        },
        {
            "text": "Skattesystemet ska vara enkelt och transparent — både "
                    "företag och privatpersoner ska tydligt se vilka skatter "
                    "som betalats. Arbetsgivaravgiften bör synliggöras genom "
                    "att den allmänna löneavgiften separeras. Då blir det "
                    "tydligt vilka skatter företagen faktiskt betalar.",
            "source": "HB01SkU15",
        },
    ],
    "L": [
        {
            "text": "Goda skattemässiga villkor för företagande är en "
                    "förutsättning för att hela landet ska växa och en viktig "
                    "del i politiken för stärkt konkurrenskraft. Ett starkt "
                    "och konkurrenskraftigt näringsliv skapar fler jobb och "
                    "säkrar välfärdens finansiering. Skattereglerna ska vara "
                    "förutsägbara.",
            "source": "HC01SkU13, HA01SkU12",
        },
        {
            "text": "Sverige stärker det internationella ramverket mot "
                    "skatteflykt. Tilläggsskatten genomför EU:s "
                    "minimibeskattningsdirektiv — vinster i stora "
                    "multinationella koncerner ska beskattas med minst 15 "
                    "procent. Det här handlar om att täppa till hål som "
                    "underminerar svenska företags konkurrens.",
            "source": "HD01SkU14",
        },
    ],
    "KD": [
        {
            "text": "Goda skattemässiga villkor för företagande är en "
                    "förutsättning för att hela landet ska växa och en viktig "
                    "del i politiken för stärkt konkurrenskraft. Ett starkt "
                    "och konkurrenskraftigt näringsliv skapar fler jobb och "
                    "säkrar välfärdens finansiering. Skattereglerna ska vara "
                    "förutsägbara.",
            "source": "HC01SkU13, HA01SkU12",
        },
        {
            "text": "Sverige stärker det internationella ramverket mot "
                    "skatteflykt. Tilläggsskatten genomför EU:s "
                    "minimibeskattningsdirektiv — vinster i stora "
                    "multinationella koncerner ska beskattas med minst 15 "
                    "procent. Det här handlar om att täppa till hål som "
                    "underminerar svenska företags konkurrens.",
            "source": "HD01SkU14",
        },
    ],
    "M": [
        {
            "text": "Goda skattemässiga villkor för företagande är en "
                    "förutsättning för att hela landet ska växa och en viktig "
                    "del i politiken för stärkt konkurrenskraft. Ett starkt "
                    "och konkurrenskraftigt näringsliv skapar fler jobb och "
                    "säkrar välfärdens finansiering. Skattereglerna ska vara "
                    "förutsägbara.",
            "source": "HC01SkU13, HA01SkU12",
        },
        {
            "text": "Sverige stärker det internationella ramverket mot "
                    "skatteflykt. Tilläggsskatten genomför EU:s "
                    "minimibeskattningsdirektiv — vinster i stora "
                    "multinationella koncerner ska beskattas med minst 15 "
                    "procent. Det här handlar om att täppa till hål som "
                    "underminerar svenska företags konkurrens.",
            "source": "HD01SkU14",
        },
    ],
    "SD": [
        {
            "text": "De flesta skattebetalare saknar tillräcklig kunskap om "
                    "vilka skatter de betalar, hur höga de är och hur de tas "
                    "ut. Det är ett demokratiskt underskott. Vi vill ha ökad "
                    "transparens — månatliga lönebesked ska visa sociala "
                    "avgifter, och det årliga slutskattebeskedet ska "
                    "pedagogiskt sammanställa allt i diagramform.",
            "source": "HB01SkU15",
        },
        {
            "text": "Sverige står för drygt en promille av världens "
                    "växthusgasutsläpp — ungefär vår andel av befolkningen. "
                    "Det är inte rimligt att Sverige utan vidare ska ta på "
                    "sig betydligt större utsläppsminskningar än andra länder. "
                    "Vår förhandlingsposition i EU och globalt ska vara att "
                    "de som släpper ut mest också gör mest.",
            "source": "HA01SkU14",
        },
    ],
}


# ============================================================================
# Topic 23 — Hälso- och sjukvård
# ============================================================================
CURATED[23] = {
    "V": [
        {
            "text": "Primärvården är allvarligt underfinansierad — dess andel "
                    "av den totala sjukvårdsbudgeten är för liten. Regeringen "
                    "bör återkomma med en nationell primärvårdsreform som "
                    "garanterar långsiktig finansiering, och statens krav på "
                    "regionerna måste följas av tillräckliga statliga resurser "
                    "för omställningen till god och nära vård.",
            "source": "HB01SoU14",
        },
        {
            "text": "Grunden för att öka antalet vårdplatser är ökad "
                    "personaltäthet. För att anställa fler och behålla de som "
                    "redan jobbar krävs inte bara mer pengar utan en reformerad "
                    "personalpolitik. Tid är det vården behöver mest av — och "
                    "det är det den får minst av.",
            "source": "HA01SoU8, HA01SoU3",
        },
    ],
    "S": [
        {
            "text": "Primärvårdsreformen går för långsamt. Mycket arbete har "
                    "bedrivits på strategisk nivå utan synliga effekter i "
                    "verksamheten. Regeringen bör öka takten — och inrätta ett "
                    "kansli vid Socialstyrelsen som följer och driver på "
                    "omställningen löpande.",
            "source": "HB01SoU14",
        },
        {
            "text": "Antalet vårdplatser behöver öka för att möta behoven. "
                    "Den förra regeringen gav Socialstyrelsen 923 miljoner "
                    "kronor att fördela för att anställa fler sjuksköterskor "
                    "och förbättra arbetsmiljön. Den typen av statligt stöd "
                    "bör permanentas — vården kan inte längre planera långsiktigt "
                    "på engångsanslag.",
            "source": "HA01SoU3",
        },
    ],
    "MP": [
        {
            "text": "Primärvården har svårt att klara sitt uppdrag och driva "
                    "omställningen till god och nära vård. Statens krav på "
                    "regionerna måste följas av tillräckliga resurser — annars "
                    "blir reformen ord på papper. Personaltätheten är "
                    "grunden för fler vårdplatser.",
            "source": "HB01SoU14, HA01SoU8",
        },
        {
            "text": "Alla regioner ska redovisa hur jämställd vården är, och "
                    "säkerställa att det finns könsuppdelad statistik. "
                    "Likarättsarbetet i vården behöver stärkas. Kvinnor söker "
                    "vård för andra symtom än män och får ofta sämre diagnos — "
                    "det är ett vårdkvalitetsproblem.",
            "source": "HA01SoU12",
        },
    ],
    "C": [
        {
            "text": "Arbetet med att korta vårdköerna måste bedrivas strukturerat, "
                    "metodiskt och systematiskt. Regeringen bör identifiera "
                    "och avskaffa hinder för att samtliga vårdgivare — oavsett "
                    "driftsform — kan bidra till att kapa köerna. Ett "
                    "övergripande mål för fler vårdplatser och nationell plan "
                    "för att nå det behövs.",
            "source": "HA01SoU3",
        },
        {
            "text": "Vården måste bli mer jämställd. Regeringen bör införa ett "
                    "genusmedicinskt uppdrag i grunduppdraget för alla "
                    "nationella programområden. Lämplig myndighet bör analysera "
                    "ojämställdhet i hälso- och sjukvården utifrån kön och "
                    "genus.",
            "source": "HA01SoU12",
        },
    ],
    "L": [
        {
            "text": "Målet med hälso- och sjukvården är god hälsa och vård på "
                    "lika villkor för hela befolkningen, med respekt för alla "
                    "människors lika värde. Den med störst behov ska ges "
                    "företräde. Vi noterar att privata sjukvårdsförsäkringar "
                    "ökat — symptom på att vården behöver utvecklas, inte att "
                    "den ska privatiseras.",
            "source": "HA01SoU5",
        },
        {
            "text": "Glesbygdens primärvård behöver stärkas inom ramen för "
                    "överenskommelsen God och nära vård 2024 — nya arbetssätt, "
                    "digitala lösningar och samverkan. Företagshälsovårdens "
                    "kompetensförsörjning är en angelägen fråga som "
                    "Myndigheten för arbetsmiljökunskap har fått ett särskilt "
                    "uppdrag att samordna.",
            "source": "HB01SoU21",
        },
    ],
    "KD": [
        {
            "text": "Målet med hälso- och sjukvården är god hälsa och vård på "
                    "lika villkor för hela befolkningen, med respekt för alla "
                    "människors lika värde. Den med störst behov ska ges "
                    "företräde. Vi noterar att privata sjukvårdsförsäkringar "
                    "ökat — symptom på att vården behöver utvecklas, inte att "
                    "den ska privatiseras.",
            "source": "HA01SoU5",
        },
        {
            "text": "Glesbygdens primärvård behöver stärkas inom ramen för "
                    "överenskommelsen God och nära vård 2024 — nya arbetssätt, "
                    "digitala lösningar och samverkan. Företagshälsovårdens "
                    "kompetensförsörjning är en angelägen fråga som "
                    "Myndigheten för arbetsmiljökunskap har fått ett särskilt "
                    "uppdrag att samordna.",
            "source": "HB01SoU21",
        },
    ],
    "M": [
        {
            "text": "Målet med hälso- och sjukvården är god hälsa och vård på "
                    "lika villkor för hela befolkningen, med respekt för alla "
                    "människors lika värde. Den med störst behov ska ges "
                    "företräde. Vi noterar att privata sjukvårdsförsäkringar "
                    "ökat — symptom på att vården behöver utvecklas, inte att "
                    "den ska privatiseras.",
            "source": "HA01SoU5",
        },
        {
            "text": "Glesbygdens primärvård behöver stärkas inom ramen för "
                    "överenskommelsen God och nära vård 2024 — nya arbetssätt, "
                    "digitala lösningar och samverkan. Företagshälsovårdens "
                    "kompetensförsörjning är en angelägen fråga som "
                    "Myndigheten för arbetsmiljökunskap har fått ett särskilt "
                    "uppdrag att samordna.",
            "source": "HB01SoU21",
        },
    ],
    "SD": [
        {
            "text": "Ett robust totalförsvar förutsätter att hälso- och "
                    "sjukvården är förberedd för kris och krig. Vi vill ha en "
                    "tydlig definition och struktur för särskilda "
                    "beredskapssjukhus — vilka som ska räknas, hur de ska "
                    "skyddas i fråga om planering, bemanning och långsiktiga "
                    "investeringar.",
            "source": "HD01SoU6",
        },
        {
            "text": "Regeringen bör ta fram en nationell strategi för "
                    "diabetesvård som minskar regionala skillnader i behandling "
                    "och tillgång till hjälpmedel. Det nationella "
                    "vårdprogrammet för palliativ vård av barn ska fullt ut "
                    "genomföras i alla regioner. Båda föräldrar ska ha "
                    "lagstadgad rätt att närvara vid förlossning.",
            "source": "HC01SoU17",
        },
    ],
}




