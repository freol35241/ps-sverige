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


# ============================================================================
# Topic 15 — Nato och utrikespolitik
# ============================================================================
CURATED[15] = {
    "V": [
        {
            "text": "Vi var emot att Sverige gick med i Nato, men medlemskapet "
                    "är nu en verklighet. Sveriges egen försvarsförmåga ska "
                    "fortfarande stå i centrum — och vi ska vara en aktiv "
                    "samarbetspartner i Norden och Östersjöområdet. Inte "
                    "operationer långt borta, utan vårt eget närområde.",
            "source": "HC01UFöU2",
        },
        {
            "text": "Sverige ska aldrig vara med och planera, förbereda eller "
                    "öva på att använda kärnvapen. Att stå bakom Natos "
                    "kärnvapenpolicy är ett åtagande som vi inte borde ha "
                    "gjort. Sverige skulle ha varit tydlig från början med "
                    "att kärnvapen inte ska tillåtas på svenskt territorium.",
            "source": "HC01UFöU1, HD01UFöU2",
        },
    ],
    "S": [
        {
            "text": "Finlands och Sveriges Natomedlemskap stärker hela "
                    "alliansen. Med båda länderna i Nato är EU- och nordiska "
                    "länder kring Östersjön bundna av samma försvarsgaranti. "
                    "Vår vision är att de nordiska och baltiska länderna nu "
                    "tillsammans tar ansvar för försvaret av vår del av "
                    "Europa.",
            "source": "HC01UFöU1",
        },
        {
            "text": "Det nordiska försvars- och säkerhetspolitiska arbetet "
                    "behöver samordnas effektivt. Vi har redan krigsplanläggning "
                    "med Finland och tätt samarbete med övriga nordiska länder "
                    "— nu bör en nordisk försvars- och säkerhetspolitisk "
                    "kommission tillsättas, med förankring i samtliga nordiska "
                    "parlament.",
            "source": "HB01UU4, HD01UU4",
        },
    ],
    "MP": [
        {
            "text": "Vi röstade emot Natoansökan men respekterar nu riksdagens "
                    "beslut. När Sverige väl är medlem är det viktigt att vi "
                    "driver en självständig utrikespolitik och står upp för "
                    "demokrati och mänskliga rättigheter — både inom Nato och "
                    "i förhållande till alliansens andra länder.",
            "source": "HC01UFöU1, HC01UU19",
        },
        {
            "text": "Solidaritet med människor i hela världen är en av våra "
                    "grundstenar. Det är beklagligt att Sverige tar emot färre "
                    "asylsökande och skär ner biståndet. Vi har ett "
                    "grundläggande medmänskligt ansvar att vara en fristad — "
                    "särskilt för dem som förföljs av totalitära regimer för "
                    "sin kamp för fri- och rättigheter.",
            "source": "HC01UU14",
        },
    ],
    "C": [
        {
            "text": "Sverige har under många år deltagit i internationella "
                    "insatser under FN-, Nato- och EU-flagg — alltid med "
                    "FN-mandat. De insatser vi deltar i gör skillnad. Den "
                    "nationella försvarsplaneringen måste säkra att vi också "
                    "kan ta vårt internationella ansvar, inte minst efter "
                    "Nato-anslutningen.",
            "source": "HC01UFöU1",
        },
        {
            "text": "Erfarenheterna av covid-19 visar att den nordiska "
                    "krisberedskapen behöver stärkas. När gränser stängdes "
                    "oberoende av varandra ledde det till förvirring som "
                    "förvärrade krisen. En genomarbetad plan för hur Norden "
                    "ska hantera framtida kriser, utan att den fria rörligheten "
                    "förloras, behövs.",
            "source": "HC01UU14, HA01UU5",
        },
    ],
    "L": [
        {
            "text": "Samarbetet mellan de nordiska länderna är ett av världens "
                    "äldsta och mest omfattande regionala samarbeten. Den "
                    "nordiska modellen bygger på hög samhällelig tillit, "
                    "starka demokrati- och folkrörelsetraditioner och djup "
                    "respekt för rättsstat, jämställdhet och mänskliga "
                    "rättigheter.",
            "source": "HA01UU5",
        },
        {
            "text": "Den svenska tillståndsprövningen av vapenexport bygger på "
                    "en helhetsbedömning utifrån regeringens riktlinjer och "
                    "etablerad praxis. ISP prövar ansökningar självständigt. "
                    "EU:s gemensamma ståndpunkt och FN:s vapenhandelsfördrag "
                    "beaktas. Det är ett ordnat, restriktivt system.",
            "source": "HC01UU9",
        },
    ],
    "KD": [
        {
            "text": "Samarbetet mellan de nordiska länderna är ett av världens "
                    "äldsta och mest omfattande regionala samarbeten. Den "
                    "nordiska modellen bygger på hög samhällelig tillit, "
                    "starka demokrati- och folkrörelsetraditioner och djup "
                    "respekt för rättsstat, jämställdhet och mänskliga "
                    "rättigheter.",
            "source": "HA01UU5",
        },
        {
            "text": "Den svenska tillståndsprövningen av vapenexport bygger på "
                    "en helhetsbedömning utifrån regeringens riktlinjer och "
                    "etablerad praxis. ISP prövar ansökningar självständigt. "
                    "EU:s gemensamma ståndpunkt och FN:s vapenhandelsfördrag "
                    "beaktas. Det är ett ordnat, restriktivt system.",
            "source": "HC01UU9",
        },
    ],
    "M": [
        {
            "text": "Samarbetet mellan de nordiska länderna är ett av världens "
                    "äldsta och mest omfattande regionala samarbeten. Den "
                    "nordiska modellen bygger på hög samhällelig tillit, "
                    "starka demokrati- och folkrörelsetraditioner och djup "
                    "respekt för rättsstat, jämställdhet och mänskliga "
                    "rättigheter.",
            "source": "HA01UU5",
        },
        {
            "text": "Den svenska tillståndsprövningen av vapenexport bygger på "
                    "en helhetsbedömning utifrån regeringens riktlinjer och "
                    "etablerad praxis. ISP prövar ansökningar självständigt. "
                    "EU:s gemensamma ståndpunkt och FN:s vapenhandelsfördrag "
                    "beaktas. Det är ett ordnat, restriktivt system.",
            "source": "HC01UU9",
        },
    ],
    "SD": [
        {
            "text": "En starkare och mer samordnad nordisk röst i EU är allt "
                    "viktigare i takt med att makt koncentreras till unionen "
                    "på bekostnad av självständiga nationalstater. Eftersom "
                    "våra nordiska länder delar både samhällssystem, "
                    "värderingar och landsgränser har vi mycket att vinna på "
                    "att samordna oss — på EU-nivå och i lagstiftningsprocessen.",
            "source": "HA01UU5",
        },
        {
            "text": "Sverige ska vara en aktiv medlem i Nato och tydligt driva "
                    "de intressen som ökar alliansens — och därmed Sveriges — "
                    "säkerhet. Alla våra nordiska grannländer är nu Nato-"
                    "medlemmar. Det är en unik historisk möjlighet att "
                    "stärka vår säkerhet i samförstånd och driva offensiv "
                    "politik som möter hot i tid, inte efter att de inträffat.",
            "source": "HC01UFöU1",
        },
    ],
}


# ============================================================================
# Topic 9 — Internationellt bistånd
# ============================================================================
CURATED[9] = {
    "V": [
        {
            "text": "I stället för att strypa biståndet bör Sverige öka stödet "
                    "till Palestina för att rädda liv. Regeringens beslut att "
                    "dra in stödet till rättighetsorganisationer i Palestina "
                    "och Israel — inklusive det ekumeniska "
                    "följeslagarprogrammet — motverkar en demokratisk och "
                    "fredlig utveckling. Civilsamhället ska stärkas, inte "
                    "försvagas.",
            "source": "HC01UU15",
        },
        {
            "text": "Kärnstöden är de viktigaste stöd Sverige kan ge till "
                    "FN-organisationerna. Det är kärnstödet som ger FN möjlighet "
                    "att vara på plats där behoven är störst och snabbt "
                    "prioritera om i akuta lägen. Regeringens ökade öronmärkning "
                    "underminerar arbetet — UN Women, UNFPA och UNAIDS har "
                    "redan drabbats.",
            "source": "HC01UU11, HA01UU3",
        },
    ],
    "S": [
        {
            "text": "Ett fredsavtal i Israel-Palestina måste bygga på en "
                    "tvåstatslösning där Israel och Palestina existerar sida "
                    "vid sida som två internationellt erkända stater. "
                    "Regeringen har valt att inte ha en särskild "
                    "Palestinastrategi och har sagt upp alla biståndsavtal "
                    "med civilsamhällesorganisationer. Vi vill att Palestina "
                    "fortsatt ska få stöd.",
            "source": "HC01UU15",
        },
        {
            "text": "I kampen om en ny världsordning är det avgörande att "
                    "Sverige och EU verkar för demokrati, mänskliga "
                    "rättigheter och hållbar utveckling — och samtidigt möter "
                    "klimatutmaningen. Vi beklagar att regeringen valt att "
                    "dra sig undan från den globala scenen. Svensk "
                    "utrikespolitik har blivit passiv.",
            "source": "HC01UU7",
        },
    ],
    "MP": [
        {
            "text": "Svenskt humanitärt bistånd till Palestina måste stärkas "
                    "för att möta de enorma behov konflikten skapat. Sverige "
                    "måste kräva att Israel som ockupationsmakt följer sina "
                    "skyldigheter. Det palestinska civilsamhället, "
                    "människorättsorganisationer och fria medier är de "
                    "aktörer som kan bygga förtroendet för en tvåstatslösning.",
            "source": "HC01UU15",
        },
        {
            "text": "Regeringen har dragit ned stödet till flera viktiga "
                    "FN-organ — strypt allt stöd till UNAIDS, minskat "
                    "finansieringen av UN Women, FN:s fredsbyggande fond, "
                    "UNFPA och UNDP. Kärnstöd är nödvändigt för att snabbt "
                    "kunna agera mot akuta kriser. Sverige har historiskt "
                    "varit ett av få länder som tagit ansvar — det är en "
                    "tradition vi inte ska överge.",
            "source": "HC01UU11",
        },
    ],
    "C": [
        {
            "text": "Kriget i Gaza har skapat ett katastrofalt mänskligt "
                    "lidande. En hållbar vapenvila är av största vikt för "
                    "att kunna leverera förnödenheter, och alla parter måste "
                    "följa folkrätten. Sverige ska fortsätta och utveckla "
                    "biståndet till Palestina, inklusive stöd till UNRWA, "
                    "som en del i det långsiktiga arbetet för en tvåstatslösning.",
            "source": "HC01UU15",
        },
        {
            "text": "När Sverige väljer att samarbeta med ett FN-organ ska "
                    "det göras utifrån en tydlig bild av att det är det mest "
                    "effektiva sättet att uppnå hållbara resultat. Det "
                    "svenska bidraget ska riktas till de FN-organ vi bedömer "
                    "fungerar väl, eller arbetar med frågor som är särskilt "
                    "viktiga för oss.",
            "source": "HA01UU3",
        },
    ],
    "L": [
        {
            "text": "Mot bakgrund av kriget i Europa ger regeringen närområdet "
                    "och stödet till Ukraina högsta prioritet i utrikes- och "
                    "säkerhetspolitiken. Den transatlantiska relationen är "
                    "fortsatt av stor vikt för Sverige och Europa — också när "
                    "den nya amerikanska administrationen skapar frågetecken "
                    "kring USA:s engagemang.",
            "source": "HC01UU7",
        },
        {
            "text": "En enad och långsiktig strategisk EU-politik gentemot "
                    "Kina är avgörande för att säkerställa svenska intressen. "
                    "Kinas snabba utveckling och stärkta roll får allt större "
                    "betydelse för svensk utrikes- och säkerhetspolitik. De "
                    "nationella svenska säkerhetsintressena ska prioriteras.",
            "source": "HC01UU7",
        },
    ],
    "KD": [
        {
            "text": "Mot bakgrund av kriget i Europa ger regeringen närområdet "
                    "och stödet till Ukraina högsta prioritet i utrikes- och "
                    "säkerhetspolitiken. Den transatlantiska relationen är "
                    "fortsatt av stor vikt för Sverige och Europa — också när "
                    "den nya amerikanska administrationen skapar frågetecken "
                    "kring USA:s engagemang.",
            "source": "HC01UU7",
        },
        {
            "text": "En enad och långsiktig strategisk EU-politik gentemot "
                    "Kina är avgörande för att säkerställa svenska intressen. "
                    "Kinas snabba utveckling och stärkta roll får allt större "
                    "betydelse för svensk utrikes- och säkerhetspolitik. De "
                    "nationella svenska säkerhetsintressena ska prioriteras.",
            "source": "HC01UU7",
        },
    ],
    "M": [
        {
            "text": "Mot bakgrund av kriget i Europa ger regeringen närområdet "
                    "och stödet till Ukraina högsta prioritet i utrikes- och "
                    "säkerhetspolitiken. Den transatlantiska relationen är "
                    "fortsatt av stor vikt för Sverige och Europa — också när "
                    "den nya amerikanska administrationen skapar frågetecken "
                    "kring USA:s engagemang.",
            "source": "HC01UU7",
        },
        {
            "text": "En enad och långsiktig strategisk EU-politik gentemot "
                    "Kina är avgörande för att säkerställa svenska intressen. "
                    "Kinas snabba utveckling och stärkta roll får allt större "
                    "betydelse för svensk utrikes- och säkerhetspolitik. De "
                    "nationella svenska säkerhetsintressena ska prioriteras.",
            "source": "HC01UU7",
        },
    ],
    "SD": [
        {
            "text": "FN har bidragit till utveckling och katastrofhjälp men "
                    "dras med stora problem: för många organisationer, för "
                    "höga overheadkostnader och återkommande "
                    "korruptionsindikationer i ekonomi, tjänstetilldelning "
                    "och landsstöd. Sverige bör verka för sammanslagningar av "
                    "FN-organ för att minska dubbelarbete och öka "
                    "kostnadseffektiviteten.",
            "source": "HA01UU3",
        },
        {
            "text": "Palestinska myndigheten visar stora brister när det gäller "
                    "korruption, yttrandefrihet, rättssäkerhet och "
                    "diskriminering av kvinnor. EU har sedan 1994 skänkt över "
                    "50 miljarder kronor till Palestina, men resultaten är "
                    "begränsade. Svenskt bistånd ska vara villkorat med "
                    "tydliga krav på reform och uppföljning.",
            "source": "HB01UU15",
        },
    ],
}


# ============================================================================
# Topic 11 — Läkemedel och tandvård
# ============================================================================
CURATED[11] = {
    "V": [
        {
            "text": "Tandvården behöver reformeras med ett verkligt "
                    "högkostnadsskydd. Kostnader över referenspriserna bör "
                    "den behandlande tandläkaren stå för, och referenspriserna "
                    "ses över för att motsvara verklig kostnad. Tandvården "
                    "är i dag en av de tydligaste klasskillnaderna i Sverige.",
            "source": "HD01SoU14",
        },
        {
            "text": "Dagens kostnadsmodeller för dyra och specialdesignade "
                    "läkemedel fungerar inte för långvariga behandlingar mot "
                    "små patientgrupper. Människor med svår, allvarlig sjukdom "
                    "riskerar att bli utan behandling eftersom den inte anses "
                    "kostnadseffektiv. Staten behöver ta ett särskilt ansvar "
                    "för dessa fall.",
            "source": "HC01SoU14",
        },
    ],
    "S": [
        {
            "text": "Ett viktigt sätt att öka tillgängligheten till god tandvård "
                    "är att sänka de ekonomiska trösklarna. Den tidigare "
                    "regeringen tillsatte en tandvårdsutredning som föreslog "
                    "en modell för nytt högkostnadsskydd. Vi anser att de "
                    "förslagen ska tas vidare — i stället för att se över "
                    "tandvårdens skydd för att efterlikna övrig vård.",
            "source": "HD01SoU14, HA01SoU15",
        },
        {
            "text": "Allas rätt till en jämlik och bra sjukvård är central. "
                    "Den grundläggande principen ska vara att vården ges "
                    "utifrån behov. För dyra läkemedel där få drabbas behöver "
                    "staten ta större ansvar — annars blir tillgång till "
                    "behandling beroende av vilken region man bor i.",
            "source": "HC01SoU14",
        },
    ],
    "MP": [
        {
            "text": "Det svenska läkemedelssystemet är inte anpassat till den "
                    "snabba utvecklingen av nya behandlingar. Hittills "
                    "gjorda utredningar har haft otillräckliga förutsättningar — "
                    "de har inte fått vara kostnadsdrivande. Det måste "
                    "ändras om vi vill att människor med sällsynta diagnoser "
                    "ska få den vård som existerar.",
            "source": "HD01SoU14",
        },
        {
            "text": "Allas rätt till en jämlik och bra sjukvård är viktig. "
                    "Vården ska ges utifrån behov. Staten bör ha ett särskilt "
                    "stöd till regionerna för behandlingsmetoder och "
                    "mediciner som är extra kostsamma — så att människor med "
                    "allvarlig sjukdom inte hamnar mellan stolarna.",
            "source": "HC01SoU14",
        },
    ],
    "C": [
        {
            "text": "Utvecklingen av nya läkemedel går snabbare än någonsin — "
                    "men det räcker inte att de utvecklas, de måste också nå "
                    "patienterna. Det svenska systemet är inte anpassat till "
                    "en situation där det finns behandling mot tillstånd som "
                    "tidigare helt saknat möjligheter. Systemet behöver ses "
                    "över.",
            "source": "HC01SoU14",
        },
        {
            "text": "Den globala läkemedelsbristen och bristen på farmaceuter "
                    "i Sverige motiverar handling. Regeringen bör ta fram en "
                    "strategi för stärkt tillgång till läkemedel i hela "
                    "landet. Särläkemedel måste bli snabbt och enkelt "
                    "tillgängliga för de patienter som behöver dem.",
            "source": "HB01SoU12",
        },
    ],
    "L": [
        {
            "text": "Att förebygga bristsituationer för läkemedel är en "
                    "angelägen fråga. Läkemedelsverket har fått ett uppdrag "
                    "att se över hur rest- och bristsituationer kan "
                    "förebyggas, och kan numera besluta om sanktionsavgifter "
                    "när företag inte uppfyller meddelandeplikten — det är "
                    "verktyg vi redan har och som ska användas.",
            "source": "HB01SoU12",
        },
        {
            "text": "Frågan om läkares förskrivningsrätt utreds. Patientsäker"
                    "hetslagens bestämmelser om återkallelse av legitimation "
                    "och begränsning av förskrivningsrätt finns redan — "
                    "regeringen utreder nu ytterligare frågor om "
                    "dokumentation, begränsningar och tillsyn av "
                    "läkemedelsförskrivning, vilket vi avvaktar.",
            "source": "HC01SoU14",
        },
    ],
    "KD": [
        {
            "text": "Att förebygga bristsituationer för läkemedel är en "
                    "angelägen fråga. Läkemedelsverket har fått ett uppdrag "
                    "att se över hur rest- och bristsituationer kan "
                    "förebyggas, och kan numera besluta om sanktionsavgifter "
                    "när företag inte uppfyller meddelandeplikten — det är "
                    "verktyg vi redan har och som ska användas.",
            "source": "HB01SoU12",
        },
        {
            "text": "Frågan om läkares förskrivningsrätt utreds. Patientsäker"
                    "hetslagens bestämmelser om återkallelse av legitimation "
                    "och begränsning av förskrivningsrätt finns redan — "
                    "regeringen utreder nu ytterligare frågor om "
                    "dokumentation, begränsningar och tillsyn av "
                    "läkemedelsförskrivning, vilket vi avvaktar.",
            "source": "HC01SoU14",
        },
    ],
    "M": [
        {
            "text": "Att förebygga bristsituationer för läkemedel är en "
                    "angelägen fråga. Läkemedelsverket har fått ett uppdrag "
                    "att se över hur rest- och bristsituationer kan "
                    "förebyggas, och kan numera besluta om sanktionsavgifter "
                    "när företag inte uppfyller meddelandeplikten — det är "
                    "verktyg vi redan har och som ska användas.",
            "source": "HB01SoU12",
        },
        {
            "text": "Frågan om läkares förskrivningsrätt utreds. Patientsäker"
                    "hetslagens bestämmelser om återkallelse av legitimation "
                    "och begränsning av förskrivningsrätt finns redan — "
                    "regeringen utreder nu ytterligare frågor om "
                    "dokumentation, begränsningar och tillsyn av "
                    "läkemedelsförskrivning, vilket vi avvaktar.",
            "source": "HC01SoU14",
        },
    ],
    "SD": [
        {
            "text": "Det är allvarligt att allt fler diabetiker inte hittar "
                    "sin medicin på apoteket. Ozempic säljs i dag utanför "
                    "förmånssystemet, vilket innebär att läkemedlet inte "
                    "primärt går till dem som mest behöver det. Regeringen "
                    "bör skyndsamt se till att diabetesmedicin inte säljs "
                    "utanför förmånssystemet.",
            "source": "HB01SoU12",
        },
        {
            "text": "Förebyggande tandvård spelar en avgörande roll. "
                    "Regelbundna kontroller och munhälsoundervisning är inte "
                    "tillräckligt utbredda — många söker vård först när "
                    "problemen blivit allvarliga, vilket gör behandlingen "
                    "dyrare och mer komplicerad. Allmänhetens kunskaper om "
                    "munhygien behöver stärkas.",
            "source": "HC01SoU14",
        },
    ],
}


# ============================================================================
# Topic 21 — Bostadspolitik
# ============================================================================
CURATED[21] = {
    "V": [
        {
            "text": "Bostaden är en grundläggande rättighet. Sverige behöver "
                    "en hållbar, långsiktig bostadspolitik som möter "
                    "hemlöshet, bostadssegregation och klimatomställning. En "
                    "bostad kan aldrig jämföras med andra handelsvaror. Vi "
                    "behöver återskapa en statlig bostadspolitik enligt "
                    "principen om generell välfärd.",
            "source": "HC01CU1, HB01CU1",
        },
        {
            "text": "Allmännyttan är en avgörande del av en social "
                    "bostadspolitik. Vid utgången av 2022 fanns 834 000 "
                    "bostäder i det allmännyttiga beståndet — 41 procent av "
                    "hyresrättsbeståndet. När Allbolagen infördes 2011 "
                    "förändrades förutsättningarna; lagen behöver ses över "
                    "för att stärka allmännyttans samhällsuppdrag.",
            "source": "HC01CU13",
        },
    ],
    "S": [
        {
            "text": "Utmaningarna på bostadsmarknaden behöver lösas av stat "
                    "och kommun i samverkan. En ny bostadsförsörjningslag "
                    "som tydliggör det gemensamma ansvaret behövs. Staten "
                    "måste ta tydligare ledning och i inledningen av varje "
                    "mandatperiod redogöra inför riksdagen för planerade "
                    "åtgärder på bostadsmarknaden.",
            "source": "HA01CU10",
        },
        {
            "text": "Byggandet av bostäder har tvärnitat — för "
                    "träbyggnadsindustrin är läget akut. Staten måste "
                    "skyndsamt bidra med aktiva åtgärder. Vi har föreslagit "
                    "statlig byggstimulans och statliga byggkrediter, och "
                    "anser att formerna för fler aktiva statliga åtgärder "
                    "inom bostadspolitiken behöver utredas.",
            "source": "HD01CU18",
        },
    ],
    "MP": [
        {
            "text": "Bostadsförsörjningslagen bör göras tvingande så att "
                    "fler kommuner lever upp till lagens krav. Att inte kunna "
                    "flytta dit man vill begränsar möjligheten att forma sitt "
                    "eget liv — och påverkar samhällsutvecklingen negativt om "
                    "människor inte kan flytta dit jobben eller utbildningen "
                    "finns.",
            "source": "HA01CU10",
        },
        {
            "text": "Hyrorna i särskilda boenden är ofta påtagligt höga, och "
                    "få äldre orkar driva en hyresnämndsprocess. Regeringen "
                    "bör se över hur ett system för rimliga hyror i "
                    "specialboenden kan utformas. Det är en viktig fråga för "
                    "att ge äldre människor förutsättningar att leva sina liv "
                    "som de vill.",
            "source": "HB01CU17",
        },
    ],
    "C": [
        {
            "text": "Allbolagen från 2011 har två bärande men motsägelsefulla "
                    "principer: allmännyttan ska både ta samhällsansvar och "
                    "vara affärsmässig. Tolkningen över tid har lett till "
                    "ökad tyngd på affärsmässigheten på bekostnad av "
                    "samhällsansvaret. Det behöver justeras tillbaka.",
            "source": "HC01CU13",
        },
        {
            "text": "På svaga marknader är det svårt att få lönsamhet i "
                    "nyproduktion. Statliga kreditgarantier behövs, men "
                    "regelverket behöver tydligare riktas mot "
                    "landsbygdskommuner. Generella investeringsstöd "
                    "snedvrider konkurrensen — i stället bör stöd för att "
                    "exempelvis få fram bostäder i hela landet utredas.",
            "source": "HD01CU18",
        },
    ],
    "L": [
        {
            "text": "Det finns ett stort behov av bostäder i landet. En väl "
                    "fungerande bostadsmarknad där konsumenternas efterfrågan "
                    "möter ett bostadsutbud är central för människors trygghet "
                    "och livskvalitet. Bostadsförsörjningslagen har redan "
                    "ändrats för att stärka det kommunala ansvaret.",
            "source": "HC01CU13",
        },
        {
            "text": "Planläggning av mark- och vattenområden är en kommunal "
                    "angelägenhet — det är ett uttryck för den kommunala "
                    "självstyrelsen som ska värnas. Samtidigt finns ett behov "
                    "av att reglerna inte i onödan försvårar planeringen. "
                    "Reformer i plan- och bygglagen ska underlätta effektiva "
                    "processer.",
            "source": "HA01CU12",
        },
    ],
    "KD": [
        {
            "text": "Det finns ett stort behov av bostäder i landet. En väl "
                    "fungerande bostadsmarknad där konsumenternas efterfrågan "
                    "möter ett bostadsutbud är central för människors trygghet "
                    "och livskvalitet. Bostadsförsörjningslagen har redan "
                    "ändrats för att stärka det kommunala ansvaret.",
            "source": "HC01CU13",
        },
        {
            "text": "Planläggning av mark- och vattenområden är en kommunal "
                    "angelägenhet — det är ett uttryck för den kommunala "
                    "självstyrelsen som ska värnas. Samtidigt finns ett behov "
                    "av att reglerna inte i onödan försvårar planeringen. "
                    "Reformer i plan- och bygglagen ska underlätta effektiva "
                    "processer.",
            "source": "HA01CU12",
        },
    ],
    "M": [
        {
            "text": "Det finns ett stort behov av bostäder i landet. En väl "
                    "fungerande bostadsmarknad där konsumenternas efterfrågan "
                    "möter ett bostadsutbud är central för människors trygghet "
                    "och livskvalitet. Bostadsförsörjningslagen har redan "
                    "ändrats för att stärka det kommunala ansvaret.",
            "source": "HC01CU13",
        },
        {
            "text": "Planläggning av mark- och vattenområden är en kommunal "
                    "angelägenhet — det är ett uttryck för den kommunala "
                    "självstyrelsen som ska värnas. Samtidigt finns ett behov "
                    "av att reglerna inte i onödan försvårar planeringen. "
                    "Reformer i plan- och bygglagen ska underlätta effektiva "
                    "processer.",
            "source": "HA01CU12",
        },
    ],
    "SD": [
        {
            "text": "Otryggheten och brottsligheten ökar i utsatta "
                    "bostadsområden. Den fysiska miljöns utformning har stor "
                    "betydelse för polisens möjligheter att bedriva effektivt "
                    "arbete. Miljonprogramsområdena är utformade så att det "
                    "är svårt för polisen att närvara utan att samtidigt "
                    "utsättas för risker — stadsplanering behöver skapa "
                    "säkrare närvaro.",
            "source": "HA01CU10",
        },
        {
            "text": "Fastighetsägare behöver utökade möjligheter att säga upp "
                    "hyresgäster som begår brott. Utredningen om tryggare "
                    "bostadsområden överlämnade sitt betänkande hösten 2023 — "
                    "regeringen har gått vidare med lagrådsremiss, och de "
                    "skarpa lagförslagen behöver införas så snart som möjligt.",
            "source": "HB01CU17",
        },
    ],
}


# ============================================================================
# Topic 24 — Funktionsnedsättning och äldreomsorg
# ============================================================================
CURATED[24] = {
    "V": [
        {
            "text": "Var tionde person på Arbetsförmedlingen får vänta över tre "
                    "år på att få sin funktionsnedsättning identifierad. "
                    "Många som har den identifierad saknar insatser eller "
                    "aktiviteter. Granskningen visar på en försämring över "
                    "tid. Att ha arbete och egen försörjning stärker "
                    "individens makt — staten har här svikit.",
            "source": "HD01AU6",
        },
        {
            "text": "Schablonbeloppet för assistansersättning har under lång tid "
                    "halkat efter kostnadsutvecklingen. Det innebär i praktiken "
                    "att kommunerna får ta en allt större del av kostnaden — "
                    "och att assistansberättigade riskerar att få sämre "
                    "assistans. Beloppet behöver räknas upp i takt med verklig "
                    "kostnadsutveckling.",
            "source": "HA01SoU21",
        },
    ],
    "S": [
        {
            "text": "Människor med svag ställning på arbetsmarknaden — inte "
                    "minst personer med funktionsnedsättningar — behöver ett "
                    "starkare och mer individanpassat stöd. Arbetsförmedlingen "
                    "ska ha sammanhållande roll, och samverkan med kommuner, "
                    "utbildningssamordnare, socialtjänst och sjukvård behöver "
                    "förbättras. Ingen ska hamna mellan stolarna.",
            "source": "HD01AU6",
        },
        {
            "text": "Hälso- och sjukvården inom äldreomsorgen behöver stärkas. "
                    "Med en åldrande befolkning ökar behoven av en samordnad "
                    "vård och omsorg. Staten behöver ta ett tydligare ansvar "
                    "för att säkerställa likvärdig kvalitet i hela landet — "
                    "och säkerställa att personalen har förutsättningar att "
                    "ge god vård.",
            "source": "HA01SoU22",
        },
    ],
    "MP": [
        {
            "text": "Tillgången till personlig assistans har försämrats — "
                    "regeringen behöver skyndsamt vidta åtgärder. Flyktingar "
                    "och skyddsbehövande ska inte diskrimineras när det gäller "
                    "möjligheten till personlig assistans. Det här är "
                    "rättighetsfrågor som inte ska bero på medborgarstatus.",
            "source": "HB01SoU13",
        },
        {
            "text": "Vid sjukhusvistelse förlorar assistansberättigade i "
                    "praktiken ofta sin ersättning, vilket kan leda till att "
                    "personliga assistenter sägs upp. Det är en orimlig effekt "
                    "av regelverket — människor som redan är i en svår "
                    "situation drabbas ytterligare. Reglerna behöver ses över.",
            "source": "HC01SoU15",
        },
    ],
    "C": [
        {
            "text": "Den som har rätt till personlig assistans ska också få "
                    "förutsättningar att delta på lika villkor i skola och "
                    "daglig verksamhet. Dagens krav på 'särskilda skäl' för "
                    "att assistansersättning ska lämnas i dessa miljöer "
                    "begränsar valmöjligheter och försämrar inlärning. "
                    "Kravet bör ses över.",
            "source": "HB01SoU13",
        },
        {
            "text": "Vid sjukhusvistelse förlorar assistansberättigade ofta "
                    "möjligheten till ersättning, även vid korta vistelser. "
                    "Det kan tvinga arbetsgivare att säga upp assistenter och "
                    "drabbar både brukare och anställda. Regelverket behöver "
                    "ses över så att kontinuitet kan upprätthållas.",
            "source": "HC01SoU15",
        },
    ],
    "L": [
        {
            "text": "Sjukförsäkringens syfte är att ge ersättning för "
                    "inkomstbortfall vid sjukdom — och stöd till individen "
                    "för återgång i arbete. Försäkringen ska både skänka "
                    "trygghet till den som inte kan arbeta och bidra till "
                    "rehabilitering. Den balansen är central för "
                    "trygghetssystemets legitimitet.",
            "source": "HC01SfU1, HA01SfU1",
        },
        {
            "text": "Regeringen arbetar för att stärka funktionshinderspolitiken "
                    "och äldreomsorgen inom existerande budgetramar. "
                    "Utgiftsområde 10 — ekonomisk trygghet vid sjukdom och "
                    "funktionsnedsättning — uppgår till cirka 123 miljarder "
                    "kronor. Nivåerna är avvägda mot Sveriges samlade "
                    "ekonomiska situation.",
            "source": "HC01SfU1",
        },
    ],
    "KD": [
        {
            "text": "Sjukförsäkringens syfte är att ge ersättning för "
                    "inkomstbortfall vid sjukdom — och stöd till individen "
                    "för återgång i arbete. Försäkringen ska både skänka "
                    "trygghet till den som inte kan arbeta och bidra till "
                    "rehabilitering. Den balansen är central för "
                    "trygghetssystemets legitimitet.",
            "source": "HC01SfU1, HA01SfU1",
        },
        {
            "text": "Regeringen arbetar för att stärka funktionshinderspolitiken "
                    "och äldreomsorgen inom existerande budgetramar. "
                    "Utgiftsområde 10 — ekonomisk trygghet vid sjukdom och "
                    "funktionsnedsättning — uppgår till cirka 123 miljarder "
                    "kronor. Nivåerna är avvägda mot Sveriges samlade "
                    "ekonomiska situation.",
            "source": "HC01SfU1",
        },
    ],
    "M": [
        {
            "text": "Sjukförsäkringens syfte är att ge ersättning för "
                    "inkomstbortfall vid sjukdom — och stöd till individen "
                    "för återgång i arbete. Försäkringen ska både skänka "
                    "trygghet till den som inte kan arbeta och bidra till "
                    "rehabilitering. Den balansen är central för "
                    "trygghetssystemets legitimitet.",
            "source": "HC01SfU1, HA01SfU1",
        },
        {
            "text": "Regeringen arbetar för att stärka funktionshinderspolitiken "
                    "och äldreomsorgen inom existerande budgetramar. "
                    "Utgiftsområde 10 — ekonomisk trygghet vid sjukdom och "
                    "funktionsnedsättning — uppgår till cirka 123 miljarder "
                    "kronor. Nivåerna är avvägda mot Sveriges samlade "
                    "ekonomiska situation.",
            "source": "HC01SfU1",
        },
    ],
    "SD": [
        {
            "text": "Att assistansberättigade inte beviljas assistansersättning "
                    "vid sjukhusvistelse kan få betydande konsekvenser — både "
                    "för den enskilde och för arbetsgivaren som kan tvingas "
                    "säga upp assistenter. Vi vill att assistansersättning ska "
                    "kunna utgå även vid längre sjukhusvistelser, så att "
                    "kontinuiteten upprätthålls.",
            "source": "HC01SoU15",
        },
        {
            "text": "Arbetsförmedlingens stöd till personer med "
                    "funktionsnedsättning fungerar inte. Riksrevisionen har "
                    "visat på allvarliga effektivitetsbrister. För människor "
                    "med funktionsnedsättning krävs ett samlat, "
                    "individanpassat stöd som faktiskt leder till arbete — "
                    "inte den nuvarande situationen där många blir kvar i "
                    "väntrum.",
            "source": "HD01AU6",
        },
    ],
}


# ============================================================================
# Topic 2 — Arbetsrätt och arbetslöshet
# ============================================================================
CURATED[2] = {
    "V": [
        {
            "text": "De senaste årens förändringar inom Arbetsförmedlingen — "
                    "digitaliseringen, nedskärningarna och reformeringen — "
                    "har påverkat arbetssökande med funktionsnedsättning "
                    "negativt. När den huvudsakliga kontaktvägen är digital "
                    "stängs människor som inte klarar den ute. "
                    "Arbetsförmedlingens fysiska närvaro behöver återställas.",
            "source": "HB01AU3",
        },
        {
            "text": "Anställningar inom vård, omsorg och välfärd ska vara "
                    "trygga och självklart heltidsanställda. Otrygga "
                    "anställningsformer, allmän visstid och staplade "
                    "tidsbegränsade jobb urholkar både den enskildes ekonomi "
                    "och välfärdens kvalitet. Den svenska modellen kräver "
                    "starka, jämbördiga parter.",
            "source": "HA01AU9",
        },
    ],
    "S": [
        {
            "text": "Arbetsförmedlingen är statens arbetsmarknadspolitiska "
                    "expertmyndighet och bör ges större handlingsfrihet att "
                    "välja individanpassade insatser. Den fysiska närvaron i "
                    "hela landet är viktig — också för att den aktiva "
                    "arbetsmarknadspolitiken ska finnas och fungera där "
                    "människor faktiskt bor och söker jobb.",
            "source": "HA01AU9",
        },
        {
            "text": "Den svenska modellen — där arbetsmarknadens parter "
                    "ansvarar för lönebildning och omställning — har gett "
                    "Sverige reallöneökningar och få konfliktdagar. Den ska "
                    "värnas. Det kräver att parterna ges utrymme att förhandla "
                    "fram lösningar, och att lagstiftaren inte tar över deras "
                    "uppgifter.",
            "source": "HB01AU3",
        },
    ],
    "MP": [
        {
            "text": "Arbetsförmedlingens internkostnader och medelsbehov för "
                    "att kunna ge personaltäta insatser åt långtidsarbetslösa "
                    "behöver utredas ordentligt. Det Statskontorets uppdrag "
                    "om förvaltningsmedel inte fullt ut kan ge svar på är: "
                    "vad krävs egentligen för att hjälpa dem som står längst "
                    "från arbetsmarknaden tillbaka in?",
            "source": "HB01AU3",
        },
        {
            "text": "Arbetsmarknadspolitiska insatser behöver vara "
                    "individanpassade. Det räcker inte med att Arbets"
                    "förmedlingen hänvisar människor till digitala lösningar. "
                    "För personer med långa vägar tillbaka in på "
                    "arbetsmarknaden krävs faktisk personalkontakt, faktisk "
                    "kompetens och faktisk tid.",
            "source": "HB01AU3",
        },
    ],
    "C": [
        {
            "text": "Det är av största vikt att bryta tudelningen på "
                    "arbetsmarknaden, motverka långtidsarbetslösheten och se "
                    "till att alla som kommer till Sverige verkligen kommer "
                    "in i samhället. Det kräver effektivt stöd och matchning. "
                    "Arbetsförmedlingens reformering behöver fortsätta — "
                    "myndigheten har inte klarat sitt ensamuppdrag.",
            "source": "HB01AU3",
        },
        {
            "text": "Den aktiva arbetsmarknadspolitiken behöver fler aktörer "
                    "och fler vägar in. Privata leverantörer, utbildnings"
                    "samordnare, socialtjänst och regionerna ska tillsammans "
                    "med Arbetsförmedlingen erbjuda individanpassade insatser. "
                    "Det viktiga är att människor får jobb — inte vilken "
                    "huvudman som står för insatsen.",
            "source": "HA01AU9",
        },
    ],
    "L": [
        {
            "text": "Den svenska modellens grundpelare är en arbetsmarknad "
                    "som underlättar omställning. Det förutsätter starka och "
                    "jämbördiga parter som samordnar lönebildning och "
                    "omställning. Parternas kompromissinriktade arbete har "
                    "gett Sverige både reallöneökningar och unikt få "
                    "konfliktdagar — det är en modell som ska värnas.",
            "source": "HB01AU3",
        },
        {
            "text": "Reformeringen av Arbetsförmedlingen fortsätter. "
                    "Myndigheten ska utvecklas mot en effektivare "
                    "matchningsroll, med fler privata leverantörer som "
                    "utför insatser. Riksdagen behöver inte gå in i detaljer "
                    "om varje insatsform — det ska Arbetsförmedlingen kunna "
                    "anpassa till behoven.",
            "source": "HB01AU8",
        },
    ],
    "KD": [
        {
            "text": "Den svenska modellens grundpelare är en arbetsmarknad "
                    "som underlättar omställning. Det förutsätter starka och "
                    "jämbördiga parter som samordnar lönebildning och "
                    "omställning. Parternas kompromissinriktade arbete har "
                    "gett Sverige både reallöneökningar och unikt få "
                    "konfliktdagar — det är en modell som ska värnas.",
            "source": "HB01AU3",
        },
        {
            "text": "Reformeringen av Arbetsförmedlingen fortsätter. "
                    "Myndigheten ska utvecklas mot en effektivare "
                    "matchningsroll, med fler privata leverantörer som "
                    "utför insatser. Riksdagen behöver inte gå in i detaljer "
                    "om varje insatsform — det ska Arbetsförmedlingen kunna "
                    "anpassa till behoven.",
            "source": "HB01AU8",
        },
    ],
    "M": [
        {
            "text": "Den svenska modellens grundpelare är en arbetsmarknad "
                    "som underlättar omställning. Det förutsätter starka och "
                    "jämbördiga parter som samordnar lönebildning och "
                    "omställning. Parternas kompromissinriktade arbete har "
                    "gett Sverige både reallöneökningar och unikt få "
                    "konfliktdagar — det är en modell som ska värnas.",
            "source": "HB01AU3",
        },
        {
            "text": "Reformeringen av Arbetsförmedlingen fortsätter. "
                    "Myndigheten ska utvecklas mot en effektivare "
                    "matchningsroll, med fler privata leverantörer som "
                    "utför insatser. Riksdagen behöver inte gå in i detaljer "
                    "om varje insatsform — det ska Arbetsförmedlingen kunna "
                    "anpassa till behoven.",
            "source": "HB01AU8",
        },
    ],
    "SD": [
        {
            "text": "I stället för subventionerade arbeten där individen "
                    "lämnar arbetsplatsen samma dag som subventionen upphör "
                    "vill vi se fler arbetsplatsnära utbildningsplatser — "
                    "användbar arbetslivserfarenhet kombinerat med teoretiska "
                    "studier. Parterna ska tillsammans utforma en svensk "
                    "modell för ett lärlingssystem.",
            "source": "HC01AU6",
        },
        {
            "text": "Subventionerade anställningar ska leda till varaktiga "
                    "jobb — inte till en svängdörr där människor slussas "
                    "vidare så snart stödet upphör. Ett svenskt "
                    "lärlingssystem byggt på parternas överenskommelser "
                    "skulle kunna ge fler vägar in på arbetsmarknaden med "
                    "långsiktig effekt.",
            "source": "HA01AU9",
        },
    ],
}










