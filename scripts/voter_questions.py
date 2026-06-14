"""Three question pools for the voter tool.

WHAT pool – policy positions, each mapped to a topic_id from the 28-cluster
analysis. `agree_direction` encodes which way "Agree" maps in stance space:
  +1: agreeing means voter aligns with the cabinet's recommendation on this
      topic (high stance value, near +1 in the party_topic matrix)
  -1: agreeing means voter aligns with the opposition position (low stance)

WHY pool – value-frame statements. The text itself is what gets SBERT-
embedded at export time. We add the embedding to the voter's reasoning
vector when they agree, subtract when they disagree.

HOW pool – mechanism-preference statements, each mapped to one of four
lexicon dimensions from the HOW analysis:
  market_vs_regulation, prevention_vs_punishment,
  state_vs_local, universal_vs_targeted
`agree_direction` is +1 if Agree aligns with the *left* side of the
dimension (market, prevention, state, universal) and -1 for the right.

Editorial note: questions are written to differentiate parties on the
intended axis. Voter-natural Swedish; no riksdag-jargon. Approximately
15 / 12 / 10 questions per pool; tool picks ~5-7 per round adaptively.
"""

WHAT_POOL = [
    {
        "id": "what_migration_strict",
        "text": "Sverige bör fortsätta ha en stram migrationspolitik och "
                "skärpta krav för medborgarskap.",
        "topic_id": 13,
        "agree_direction": +1,  # cabinet-aligned
    },
    {
        "id": "what_migration_asylum",
        "text": "Asylrätten ska upprätthållas även när invandringen är hög.",
        "topic_id": 13,
        "agree_direction": -1,  # opposition-aligned (V, MP)
    },
    {
        "id": "what_migration_deport",
        "text": "Sverige bör snabbt verkställa fler utvisningar av personer "
                "som saknar uppehållstillstånd.",
        "topic_id": 13,
        "agree_direction": +1,
    },
    {
        "id": "what_klimat_nuclear",
        "text": "Sverige bör bygga ny kärnkraft för att klara elförsörjningen.",
        "topic_id": 18,
        "agree_direction": +1,  # cabinet line
    },
    {
        "id": "what_klimat_fuel_tax",
        "text": "Drivmedelsskatten bör sänkas, även om utsläppen ökar något.",
        "topic_id": 18,
        "agree_direction": +1,  # cabinet+SD position on reduktionsplikten
    },
    {
        "id": "what_klimat_radical",
        "text": "Klimatomställningen måste gå snabbare, även om den kostar "
                "mer på kort sikt.",
        "topic_id": 18,
        "agree_direction": -1,  # opposition (V, MP, S)
    },
    {
        "id": "what_skola_profit",
        "text": "Vinstdrivande friskolor bör förbjudas eller starkt begränsas.",
        "topic_id": 7,
        "agree_direction": -1,  # V, MP
    },
    {
        "id": "what_skola_state",
        "text": "Staten bör ta över huvudansvaret för svensk skola från "
                "kommunerna.",
        "topic_id": 7,
        "agree_direction": -1,  # S, V, MP
    },
    {
        "id": "what_brott_punishment",
        "text": "Straffen för grov brottslighet bör skärpas kraftigt – även "
                "livstid utan möjlighet till omvandling.",
        "topic_id": 3,
        "agree_direction": +1,  # cabinet+SD
    },
    {
        "id": "what_brott_prevention",
        "text": "Förebyggande sociala insatser är viktigare än hårdare straff "
                "för att minska brottsligheten.",
        "topic_id": 3,
        "agree_direction": -1,  # V, MP, S
    },
    {
        "id": "what_skatt_lower",
        "text": "Skatten på arbete bör sänkas – det skapar fler jobb i "
                "längden.",
        "topic_id": 6,
        "agree_direction": +1,  # cabinet
    },
    {
        "id": "what_skatt_rich",
        "text": "Höginkomsttagare bör bidra mer för att finansiera "
                "försvarsupprustningen och välfärden.",
        "topic_id": 6,
        "agree_direction": -1,  # V, S
    },
    {
        "id": "what_vard_private",
        "text": "Privata sjukvårdsförsäkringar bör begränsas – vård ska ges "
                "på lika villkor genom det offentliga.",
        "topic_id": 23,
        "agree_direction": -1,  # V, MP, S
    },
    {
        "id": "what_bistand_keep",
        "text": "Sverige bör behålla utvecklingsbiståndet på en hög nivå, "
                "även när vår ekonomi är pressad.",
        "topic_id": 9,
        "agree_direction": -1,  # V, MP, S
    },
    {
        "id": "what_nato_full",
        "text": "Sverige ska fullt ut delta i Natos säkerhetsgemenskap, "
                "inklusive alliansens kärnvapenstrategi.",
        "topic_id": 15,
        "agree_direction": +1,  # cabinet+SD
    },
]


WHY_POOL = [
    {
        "id": "why_security_freedom",
        "text": "Trygghet i samhället är viktigare än individuell frihet.",
    },
    {
        "id": "why_climate_primary",
        "text": "Klimatet är vår tids mest avgörande politiska fråga.",
    },
    {
        "id": "why_state_welfare",
        "text": "Det är staten som ska bära huvudansvaret för välfärden – "
                "inte familjen eller civilsamhället.",
    },
    {
        "id": "why_national_identity",
        "text": "Sverige bör värna sin nationella identitet och kulturella "
                "sammanhållning.",
    },
    {
        "id": "why_market_works",
        "text": "Marknaden löser många samhällsproblem bättre än politiska "
                "beslut.",
    },
    {
        "id": "why_universal_rights",
        "text": "Människor i nöd har rätt till skydd, oavsett vart de "
                "kommer ifrån.",
    },
    {
        "id": "why_individual_freedom",
        "text": "Det viktigaste är att människor får möjlighet att forma "
                "sitt eget liv.",
    },
    {
        "id": "why_sweden_leads",
        "text": "Sveriges roll i världen är att gå före – i klimat, fred och "
                "mänskliga rättigheter.",
    },
    {
        "id": "why_tradition",
        "text": "Politiken bör utgå från traditioner och pröva förändringar "
                "försiktigt, snarare än att riva upp för att bygga nytt.",
    },
    {
        "id": "why_equal_outcome",
        "text": "Jämlikhet i utfall är viktigare än enbart lika "
                "möjligheter – politik ska aktivt utjämna skillnader.",
    },
    {
        "id": "why_order",
        "text": "Lag och ordning måste prioriteras framför sociala "
                "förklaringar till samhällsproblem.",
    },
    {
        "id": "why_solidarity",
        "text": "Människovärdet är okränkbart – det innebär att vi har "
                "ansvar för dem som har minst, även när det kostar.",
    },
]


HOW_POOL = [
    {
        "id": "how_market_taxes",
        "text": "Skattesänkningar är ofta bättre än ökade offentliga utgifter.",
        "dimension": "market_vs_regulation",
        "agree_direction": +1,  # market
    },
    {
        "id": "how_market_private",
        "text": "Privata aktörer gör välfärden mer effektiv.",
        "dimension": "market_vs_regulation",
        "agree_direction": +1,
    },
    {
        "id": "how_regulation_market",
        "text": "Marknaden behöver hårdare regleringar för att fungera "
                "rättvist.",
        "dimension": "market_vs_regulation",
        "agree_direction": -1,  # regulation
    },
    {
        "id": "how_prevention",
        "text": "Förebyggande sociala insatser är mer effektiva än skärpta "
                "straff.",
        "dimension": "prevention_vs_punishment",
        "agree_direction": +1,  # prevention
    },
    {
        "id": "how_punishment",
        "text": "Hårdare straff är det effektivaste sättet att minska "
                "kriminaliteten.",
        "dimension": "prevention_vs_punishment",
        "agree_direction": -1,  # punishment
    },
    {
        "id": "how_local",
        "text": "Politiska beslut bör fattas så lokalt som möjligt – "
                "kommunerna bättre än staten.",
        "dimension": "state_vs_local",
        "agree_direction": -1,  # local
    },
    {
        "id": "how_state",
        "text": "Staten behöver ta större ansvar för att skapa nationell "
                "likvärdighet i välfärden.",
        "dimension": "state_vs_local",
        "agree_direction": +1,  # state
    },
    {
        "id": "how_universal",
        "text": "Välfärden ska vara generell – alla får samma, så bidragen "
                "inte stigmatiserar.",
        "dimension": "universal_vs_targeted",
        "agree_direction": +1,  # universal
    },
    {
        "id": "how_targeted",
        "text": "Stöd ska gå till dem som behöver mest – inte slösas på de "
                "som klarar sig själva.",
        "dimension": "universal_vs_targeted",
        "agree_direction": -1,  # targeted
    },
    {
        "id": "how_civil_society",
        "text": "Civilsamhället och föreningslivet bör göra mer av det som "
                "staten gör i dag.",
        "dimension": "state_vs_local",
        "agree_direction": -1,
    },
]
