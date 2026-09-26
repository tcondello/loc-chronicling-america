"""Curated historical story chapters contrasting regional broadsheets (1890–1910).

Designed for the Library of Congress Chronicling America & Pinecone showcase.
Curated into the definitive Top 5 viral story chapters across:
- Los Angeles Herald (West Coast, Southern California - sn85042462)
- The San Francisco Call (West Coast, Bay Area - sn85066387)
- The Evening World (East Coast, New York - sn83030193)
- The Sun (East Coast, New York - sn83030272)
- Chicago Eagle (Midwest, Great Lakes - sn84025828)
- The Beatrice Daily Express (Midwest & Heartland, Great Plains - sn84020107)
"""

from typing import Any, Dict, List

TARGET_YEARS = ["1890", "1891", "1892", "1905", "1906", "1907", "1908", "1909", "1910"]

CHAPTERS: List[Dict[str, Any]] = [
    {
        "id": "chapter_1",
        "number": 1,
        "badge": "🗽 1891–1892 • Two Gates of the Republic",
        "title": "Two Gates of the Republic: The Golden Door vs. The Iron Wall",
        "subtitle": "Ellis Island's Grand Opening vs. California's 'Dog-Tag' Geary Act of 1892",
        "period": "1891–1892",
        "years": ["1891", "1892"],
        "narrative": (
            "On New Year's Day 1892, seventeen-year-old Irish immigrant Annie Moore stepped ashore as the first arrival registered "
            "at the newly opened federal Ellis Island station in New York Harbor, as the metropolis welcomed steamships fleeing European "
            "famine and pogroms. Yet in the exact same legislative session, California lawmakers drove the passage of the Geary Act of 1892. "
            "This statute mandated that all Chinese residents carry internal resident certificates ('dog-tag passports') with photographic "
            "identification—creating America's first domestic photo-ID regime backed by federal marshals, border patrols, and deportation under hard labor."
        ),
        "west_angle": "🌴 Los Angeles Herald: Tracked federal marshals patrolling the Mexican border to intercept Chinese arrivals and reported courtroom habeas corpus battles over certificate verification.",
        "east_angle": "☀️ The Sun: Chronicled transatlantic steamship arrivals at Ellis Island, immigrant processing throughput, and New York harbor welcoming ceremonies.",
        "midwest_angle": "🌾 The Beatrice Daily Express: Reported official diplomatic protests from China to Secretary of State Blaine alongside congressional debates on immigration restrictions.",
        "figures": ["Annie Moore", "Thomas J. Geary", "John B. Weber", "James G. Blaine", "Federal Border Marshals"],
        "query": '+(immigrant OR immigration OR Chinese OR "Ellis Island" OR exclusion OR passport OR "Geary Act" OR marshal)',
        "semantic_query": "Ellis Island grand opening immigration contrasted with Chinese exclusion enforcement, border marshals, and certificate passports in California",
        "default_newspapers": ["los_angeles_herald", "the_sun", "the_beatrice_daily_express"],
        "mode": "Hybrid (Dual-Path RRF)",
        "operator_notes": "Pairs required group `+(immigrant OR immigration OR Chinese OR \"Ellis Island\" OR exclusion OR passport OR \"Geary Act\" OR marshal)` to capture both European entry and Pacific border enforcement simultaneously.",
    },
    {
        "id": "chapter_2",
        "number": 2,
        "badge": "🪓 1892 • Primordial True Crime",
        "title": "Forty Whacks: Blood, Morphine & The Fall River Axe Murders",
        "subtitle": "Andrew Borden Butchered, Poisoned Mutton & The Arrest of Lizzie Borden",
        "period": "1892",
        "years": ["1892"],
        "narrative": (
            "On a sweltering August morning in 1892, wealthy Fall River banker Andrew Borden and his wife Abby were butchered "
            "inside their locked Massachusetts home by dozens of savage hatchet blows that crushed their skulls. "
            "When their prim, 32-year-old Sunday-school-teaching daughter Lizzie was arrested following bizarre inquest testimony, "
            "suspicious dress-burnings, and inquiries into prussic acid, Gilded Age America lost its collective mind in the nation's "
            "very first viral true-crime tabloid media circus."
        ),
        "west_angle": "🌴 Los Angeles Herald: Relayed coast-to-coast telegraph dispatches, debating whether patrician New England morality concealed savage domestic murder.",
        "east_angle": "🗽 The Evening World: Turned into a daily crime magazine with front-page inquest reports, coroner court proceedings, and morbid forensic speculation.",
        "midwest_angle": "🌾 The Beatrice Daily Express: Published rapid daily bulletins on the arrest of Miss Lizzie Borden, eyewitness reports of neighbors jumping fences, and physician testimony.",
        "figures": ["Lizzie Borden", "Andrew J. Borden", "Abby Borden", "Dr. H. Bowen", "Bridget Sullivan"],
        "query": '+(Borden OR "Lizzie Borden") +(murder OR axe OR hatchet OR inquest OR trial OR "Fall River" OR arrest)',
        "semantic_query": "The Fall River Massachusetts axe murders of Andrew and Abby Borden, and the inquest, dress burning, and arrest of Lizzie Borden",
        "default_newspapers": ["the_beatrice_daily_express", "the_evening_world", "los_angeles_herald"],
        "mode": "Hybrid (Dual-Path RRF)",
        "operator_notes": "Uses mandatory name group `+(Borden OR \"Lizzie Borden\")` joined with crime terminology `+(murder OR axe OR hatchet OR inquest OR trial OR \"Fall River\" OR arrest)` to cleanly isolate murder reporting.",
    },
    {
        "id": "chapter_3",
        "number": 3,
        "badge": "🔥 1906–1907 • Cataclysm & Contagion",
        "title": "Cataclysm & Contagion: The San Francisco Inferno & Panic of 1907",
        "subtitle": "San Andreas Rupture & Firestorm to the Knickerbocker Trust Bank Run",
        "period": "1906–1907",
        "years": ["1906", "1907"],
        "narrative": (
            "At 5:12 AM on April 18, 1906, the San Andreas fault ruptured, devastating San Francisco before an uncontrollable "
            "three-day firestorm consumed 500 city blocks and incinerated its newspaper publishing plants. "
            "The physical catastrophe forced British and European underwriters to pay over $100 million in gold insurance claims, "
            "draining liquidity from Wall Street. This transatlantic contraction sparked the Panic of 1907 eighteen months later, "
            "causing terrified depositors to mob the Knickerbocker Trust in New York and forcing J. Pierpont Morgan into late-night emergency bailouts."
        ),
        "west_angle": "🌁 The San Francisco Call: Reported from the ruins of its incinerated tower, chronicling citywide reconstruction, emergency martial orders, and massive insurance proof of loss litigation.",
        "east_angle": "🗽 The Evening World: Documented Wall Street liquidity drains, depositors besieging Manhattan trust institutions, and midnight rescue conferences.",
        "midwest_angle": "🦅 Chicago Eagle: Tracked Midwestern financial fallout, bank clearing-house scrip, seismic reports, and relief freight movements across the continent.",
        "figures": ["J. Pierpont Morgan", "John D. Spreckels", "Mayor Eugene Schmitz", "General Frederick Funston", "Knickerbocker Depositors"],
        "query": '+("San Francisco" OR "Wall Street") +(earthquake OR catastrophe OR fire OR ruins OR relief OR panic OR "bank run" OR Knickerbocker) -patent -remedy',
        "semantic_query": "San Francisco earthquake catastrophe and firestorm damage alongside Wall Street financial panic of 1907 and bank runs",
        "default_newspapers": ["the_san_francisco_call", "los_angeles_herald", "chicago_eagle"],
        "mode": "Hybrid (Dual-Path RRF)",
        "operator_notes": "Combines geographic anchors `+(\"San Francisco\" OR \"Wall Street\")` with disaster and panic keywords while filtering commercial noise with `-patent -remedy`.",
    },
    {
        "id": "chapter_4",
        "number": 4,
        "badge": "✈️ 1909–1910 • The Dawn of Flight",
        "title": "Wings Over the Metropolis: Governors Island to Dominguez Field",
        "subtitle": "Wilbur Circling Lady Liberty vs. America's First Aviation Meet in Los Angeles",
        "period": "1909–1910",
        "years": ["1909", "1910"],
        "narrative": (
            "In October 1909, Wilbur Wright took off from Governors Island with a red canoe strapped beneath his biplane "
            "and soared up the Hudson River circling the Statue of Liberty before one million stunned New Yorkers. "
            "Just three months later, in January 1910, Los Angeles hosted America's very first International Air Meet at Dominguez Field. "
            "Tens of thousands watched French aviator Louis Paulhan smash the world altitude record over eucalyptus hills, "
            "Glenn Curtiss set passenger speed records, and dirigibles battle the Pacific winds—establishing Southern California as the aerospace capital of the world."
        ),
        "west_angle": "🌴 Los Angeles Herald: Published daily front pages and diagrams of the Dominguez Field meet, tracking Clifford Harmon's $5,000 balloon and Paulhan's altitude conquests.",
        "east_angle": "☀️ The Sun: Hailed the Wright brothers' record-breaking flights, Michelin trophy endurance records, and the military potential of aeroplanes.",
        "midwest_angle": "🌁 The San Francisco Call: Chronicled aerial daredevils Lincoln Beachey and Roy Knabenshue assembling dirigibles and surveyed the conquest of the air.",
        "figures": ["Wilbur Wright", "Orville Wright", "Louis Paulhan", "Glenn Curtiss", "Roy Knabenshue", "Lincoln Beachey"],
        "query": '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane OR dirigible OR balloon) AND (flight OR aviation OR altitude OR Dominguez OR Wright OR Paulhan OR Beachey OR Knabenshue)',
        "semantic_query": "Wilbur Wright flying over New York harbor and the historic first Los Angeles International Air Meet at Dominguez Field with Louis Paulhan and Glenn Curtiss",
        "default_newspapers": ["los_angeles_herald", "the_san_francisco_call", "the_sun"],
        "mode": "Hybrid (Dual-Path RRF)",
        "operator_notes": "Applies term boosting on early aviation terminology (`aeroplane^3`, `\"flying machine\"^2`) joined with aviation metrics and pioneer pilot names.",
    },
    {
        "id": "chapter_5",
        "number": 5,
        "badge": "☄️ 1910 • Cosmic Terror & Satire",
        "title": "Cosmic Terror & Cyanogen Gas: The Halley's Comet Panic of 1910",
        "subtitle": "Poisonous Cyanogen Hysteria, Rooftop Watch Parties & Mount Wilson Telescopes",
        "period": "1910",
        "years": ["1910"],
        "narrative": (
            "In May 1910, Halley's Comet swept past Earth, with astronomers warning that the planet would pass directly through "
            "its million-mile tail of toxic cyanogen gas. Sensational headlines triggered a worldwide panic: opportunists sold 'anti-comet pills' "
            "and oxygen masks, terrified citizens sealed doors with wax, and rooftop watch parties braced for doom. "
            "When the rendezvous passed harmlessly, broadsheets erupted in cynical satire. Above Los Angeles, astronomers at the "
            "Mount Wilson Observatory used their 60-inch telescope to capture pioneering spectrographic images of the celestial intruder."
        ),
        "west_angle": "🌴 Los Angeles Herald: Published celestial tracking reports and spectrographic findings from astronomers atop Mount Wilson.",
        "east_angle": "🌁 The San Francisco Call: Documented seismic and atmospheric observations as thousands gathered on Bay Area hills to watch the skies.",
        "midwest_angle": "🦅 Chicago Eagle: Published sharp satirical post-mortems—declaring: 'Said the comet to the earth: \"Never touched me.\" And the earth replied in kind'—while tracking panic suicides.",
        "figures": ["George Ellery Hale", "Camille Flammarion", "Mount Wilson Astronomers", "Chicago Satirists"],
        "query": '("Halley"~2 OR "comet"~1) AND (tail OR cyanogen OR terror OR panic OR gas OR astronomer OR sky OR "Mount Wilson" OR "Never touched me")',
        "semantic_query": "Astronomical observation of Halley comet cyanogen gas tail at Mount Wilson and mass hysteria, gas masks, and rooftop panic in 1910",
        "default_newspapers": ["chicago_eagle", "los_angeles_herald", "the_san_francisco_call"],
        "mode": "Hybrid (Dual-Path RRF)",
        "operator_notes": "Uses proximity slop `\"Halley\"~2` and `\"comet\"~1` joined with hysteria keywords and the famous headline quote 'Never touched me'.",
    },
]

# Quick mapping by ID
CHAPTER_MAP: Dict[str, Dict[str, Any]] = {c["id"]: c for c in CHAPTERS}
