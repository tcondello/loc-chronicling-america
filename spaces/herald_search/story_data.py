"""Curated historical story chapters contrasting Los Angeles and New York City (1890–1910).

Designed for the Library of Congress Chronicling America & Pinecone showcase.
Features 7 narrative chapters across:
- Los Angeles Herald (West Coast, California - sn85042462)
- The Evening World (East Coast, New York - sn83030193)
"""

from typing import Any, Dict, List

TARGET_YEARS = ["1890", "1891", "1892", "1905", "1906", "1907", "1908", "1909", "1910"]

CHAPTERS: List[Dict[str, Any]] = [
    {
        "id": "chapter_1",
        "number": 1,
        "badge": "💰 1890–1892 • Frontier vs. Wall Street",
        "title": "Gold Vaults, Silver Mines & The Broken Frontier",
        "subtitle": "Wall Street Orthodoxy, Populist Free Silver & The 1892 LA Oil Discovery",
        "period": "1890–1892",
        "years": ["1890", "1891", "1892"],
        "narrative": (
            "In 1890, the U.S. Census declared the American Western frontier officially closed. "
            "Yet in California, Los Angeles was just beginning to awaken from a sleepy pueblo into a modern boomtown. "
            "In the autumn of 1892, Edward L. Doheny and Charles A. Canfield struck oil near Second Street and Glendale "
            "Boulevard using an improvised wooden rig and a greasy eucalyptus trunk, igniting the great California petroleum boom. "
            "Simultaneously in New York City, Wall Street financial titans and Grover Cleveland's Treasury guarded the gold standard "
            "against the insurgent Populist Free Silver movement sweeping western agrarian states."
        ),
        "west_angle": "🌴 Los Angeles Herald: Cheered western mining strikes, free silver coinage, and the discovery of local oil pools transforming Southern California.",
        "east_angle": "🗽 The Evening World: Defended gold reserves, analyzed European bullion flows, and reported bank panics threatening Wall Street liquidity.",
        "figures": ["Edward L. Doheny", "Grover Cleveland", "J. Pierpont Morgan", "Charles Canfield"],
        "query": '+(currency OR silver OR "Wall Street") +(gold OR panic OR treasury OR oil)',
        "semantic_query": "Monetary currency debates between western silver mining and Wall Street gold reserves, alongside early oil discoveries",
        "mode": "Query string (Lucene)",
        "operator_notes": "Combines required group `+(currency OR silver OR \"Wall Street\")` with thematic keywords `+(gold OR panic OR treasury OR oil)` to isolate macroeconomic and resource battles.",
    },
    {
        "id": "chapter_2",
        "number": 2,
        "badge": "🗽 1891–1892 • Two Gates of the Republic",
        "title": "Two Gates of the Republic: The Golden Door vs. The Iron Wall",
        "subtitle": "Ellis Island's Grand Opening vs. California's Geary Act of 1892",
        "period": "1891–1892",
        "years": ["1891", "1892"],
        "narrative": (
            "On January 1, 1892, seventeen-year-old Annie Moore of County Cork, Ireland, became the first immigrant "
            "registered at the newly opened federal Ellis Island station in New York Harbor, as the metropolis welcomed hundreds "
            "of thousands fleeing European famine and pogroms. "
            "Yet in the exact same legislative session, California Congressman Thomas J. Geary authored the Geary Act of 1892. "
            "This law required Chinese residents in California and Los Angeles to carry internal resident certificates ('dog-tag passports') "
            "bearing photographic identification—creating America's first domestic photo-ID regime under penalty of hard labor and deportation."
        ),
        "west_angle": "🌴 Los Angeles Herald: Documented intense federal enforcement of Chinese exclusion, local Chinatown raids, and certificate verification.",
        "east_angle": "🗽 The Evening World: Celebrated transatlantic steamship arrivals, Ellis Island processing throughput, and Tammany Hall naturalization rallies.",
        "figures": ["Annie Moore", "Thomas J. Geary", "John B. Weber", "Geary Act Inspectors"],
        "query": '+(immigrant OR immigration OR Chinese OR "Ellis Island" OR exclusion OR passport)',
        "semantic_query": "Immigration processing at Ellis Island compared with Chinese exclusion enforcement and certificate passports in California",
        "mode": "Query string (Lucene)",
        "operator_notes": "Required term group matches both European arrival terminology and California exclusion vocabulary simultaneously.",
    },
    {
        "id": "chapter_3",
        "number": 3,
        "badge": "💥 1890–1892 & 1910 • Capital & Labor",
        "title": "Capital, Dynamite & The Open Shop Crucible",
        "subtitle": "The Homestead Steel Clash to the 1910 Los Angeles Times Bombing",
        "period": "1890–1892 & 1910",
        "years": ["1890", "1891", "1892", "1910"],
        "narrative": (
            "During the summer of 1892, Carnegie Steel Chairman Henry Clay Frick locked out union steelworkers at Homestead, "
            "sending Pinkerton armed guards up the Monongahela River, leading to pitched gun battles and Frick's shooting by anarchist Alexander Berkman—news "
            "that gripped New York front pages. "
            "Eighteen years later in Los Angeles, General Harrison Gray Otis, publisher of the Los Angeles Times and arch-crusader for the "
            "'Open Shop' anti-union city, saw his fortress newspaper plant dynamited at 1:07 AM on October 1, 1910, killing 21 workers. "
            "The trial of the McNamara brothers became the nationwide labor trial of the century, covered breathlessly in Manhattan."
        ),
        "west_angle": "🌴 Los Angeles Herald: Covered the tragic Times explosion, national manhunts by detective William J. Burns, and the citywide open-shop battle.",
        "east_angle": "🗽 The Evening World: Dispatched star correspondents to LA, contrasting Western industrial warfare with East Coast union movements.",
        "figures": ["Harrison Gray Otis", "Henry Clay Frick", "Alexander Berkman", "William J. Burns", "John J. McNamara"],
        "query": '+(strike OR union OR dynamite OR labor OR "open shop" OR explosion)',
        "semantic_query": "Labor union unrest, dynamite explosions, industrial strikes, and the open shop conflict between workers and capital",
        "mode": "Query string (Lucene)",
        "operator_notes": "Captures violent industrial conflicts using boolean OR disjunction inside a required group `+(...)` to track strikes and dynamite conspiracies.",
    },
    {
        "id": "chapter_4",
        "number": 4,
        "badge": "🔥 1906–1907 • Cataclysm & Panic",
        "title": "Cataclysm & Liquidity: The 1906 Earthquake & Panic of 1907",
        "subtitle": "Southern California Relief Trains vs. Wall Street's Knickerbocker Bank Run",
        "period": "1906–1907",
        "years": ["1906", "1907"],
        "narrative": (
            "At 5:12 AM on April 18, 1906, the San Andreas fault ruptured, devastating San Francisco. "
            "Within hours, Los Angeles mobilized emergency relief trains loaded with food, water, and medical staff rushing up the Central Valley. "
            "In New York, British insurance companies suffered staggering $100M+ underwriting losses, prompting London banks to drain gold bullion from Wall Street. "
            "This liquidity contraction directly triggered the Panic of 1907 eighteen months later, causing the sudden collapse of New York's Knickerbocker Trust "
            "and forcing Los Angeles banks to issue emergency clearing-house scrip to prevent total collapse."
        ),
        "west_angle": "🌴 Los Angeles Herald: Tracked continuous relief trains to the ruined north, severed telegraphs, and later the adoption of clearing-house scrip.",
        "east_angle": "🗽 The Evening World: Detailed Wall Street bank runs, queues outside Knickerbocker Trust, and J. Pierpont Morgan's dramatic late-night bailout.",
        "figures": ["J. Pierpont Morgan", "Mayor Schmitz", "Knickerbocker Depositors", "LA Clearing House"],
        "query": '+"San Francisco" +(earthquake OR catastrophe OR fire OR relief OR panic OR "bank run") -patent -remedy',
        "semantic_query": "San Francisco earthquake destruction and emergency relief alongside the Wall Street financial panic of 1907",
        "mode": "Query string (Lucene)",
        "operator_notes": "Uses `+\"San Francisco\"` mandatory exact phrase while excluding commercial noise `-patent -remedy` to isolate disaster and panic reporting.",
    },
    {
        "id": "chapter_5",
        "number": 5,
        "badge": "🚗 1905–1908 • Infrastructure & Velocity",
        "title": "Empires of Asphalt & Water: The Model T, Good Roads & Aqueducts",
        "subtitle": "Mulholland's Owens River Vision vs. Catskill Reservoirs & The Great New York-to-Paris Race",
        "period": "1905–1908",
        "years": ["1905", "1906", "1907", "1908"],
        "narrative": (
            "Between 1905 and 1908, America reinvented the physical fabric of urban civilization. "
            "In Southern California, William Mulholland surveyed the 233-mile Los Angeles Aqueduct from the Owens Valley, declaring "
            "'There it is, take it!' as the desert city secured the water needed for a boundless future. "
            "Simultaneously, New York broke ground on the monumental Ashokan Reservoir and Catskill Aqueduct. "
            "Meanwhile on the surface, Henry Ford released the Model T in October 1908, and in February 1908, 250,000 New Yorkers crammed Times Square "
            "to watch the start of the historic New York-to-Paris Automobile Race across unpaved continents."
        ),
        "west_angle": "🌴 Los Angeles Herald: Promoted the municipal water bonds, Owens River construction surveys, and the 'Good Roads' automobile revolution.",
        "east_angle": "🗽 The Evening World: Chronicled the Great Auto Race through blizzards, Catskill aqueduct engineering contracts, and New York traffic regulations.",
        "figures": ["William Mulholland", "Henry Ford", "George B. McClellan Jr.", "Montague Roberts"],
        "query": '+(aqueduct OR automobile OR "motor car" OR "good roads" OR water OR reservoir)',
        "semantic_query": "Civil engineering aqueducts, municipal water supply, automotive touring races, and good roads movement",
        "mode": "Query string (Lucene)",
        "operator_notes": "Captures dual transformation of civil hydraulics (aqueduct, reservoir) and automotive transport (\"motor car\", \"good roads\").",
    },
    {
        "id": "chapter_6",
        "number": 6,
        "badge": "✈️ 1909–1910 • The Dawn of Flight",
        "title": "Wings Over the Metropolis: Governors Island to Dominguez Field",
        "subtitle": "Wilbur Wright Circling Lady Liberty vs. America's First Aviation Meet in Los Angeles",
        "period": "1909–1910",
        "years": ["1909", "1910"],
        "narrative": (
            "In early October 1909, during the Hudson-Fulton Celebration, Wilbur Wright took off from Governors Island and flew up the Hudson River, "
            "circling the Statue of Liberty with a red canoe strapped beneath his biplane in case of water landing—witnessed by over one million New Yorkers. "
            "Just three months later, from January 10 to 20, 1910, Los Angeles hosted the Los Angeles International Air Meet at Dominguez Field—the very first "
            "major aviation competition in American history. French aviator Louis Paulhan smashed the world altitude record (4,164 ft), and Glenn Curtiss set "
            "world passenger speed records under sunny Southern California skies, establishing LA as the aerospace capital of the world."
        ),
        "west_angle": "🌴 Los Angeles Herald: Devoted daily front pages with giant photographs and diagrams to the historic flights at Dominguez Field.",
        "east_angle": "🗽 The Evening World: Marvelled at Wilbur Wright's aerial conquest of New York Harbor and tracked the European and Californian air meets.",
        "figures": ["Wilbur Wright", "Orville Wright", "Glenn Curtiss", "Louis Paulhan", "Roy Knabenshue"],
        "query": '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane) AND (flight OR speed OR altitude OR meet)',
        "semantic_query": "Early aviation pioneers, biplanes, monoplanes, world altitude records, and air meets in New York and Los Angeles",
        "mode": "Query string (Lucene)",
        "operator_notes": "Demonstrates term boosting (`aeroplane^3`, `\"flying machine\"^2`) joined with boolean AND disjunction for aviation metrics.",
    },
    {
        "id": "chapter_7",
        "number": 7,
        "badge": "☄️ 1910 • Science & Democracy",
        "title": "Cosmic Skies & Ballot Boxes: Halley's Comet & The Battle for the Ballot",
        "subtitle": "Mount Wilson's Cyanogen Spectrographs vs. Tammany Hall & California's Progressive Triumph",
        "period": "1910",
        "years": ["1910"],
        "narrative": (
            "In May 1910, Halley's Comet swept past Earth. In New York, sensationalist headlines triggered panic, gas mask sales, and rooftop watch parties. "
            "Above Los Angeles, astronomers at the Mount Wilson Observatory used their cutting-edge 60-inch reflector telescope to analyze the comet's cyanogen gas tail, "
            "putting California at the pinnacle of global astrophysics. "
            "At the same moment down on earth, American democracy was in revolt: California elected reform governor Hiram Johnson, sweeping aside the Southern Pacific Railroad "
            "political machine and preparing for California's landmark 1911 Equal Suffrage constitutional amendment—while New York suffragists fought the entrenched Tammany machine."
        ),
        "west_angle": "🌴 Los Angeles Herald: Published Mount Wilson astronomical findings, progressive Republican election bulletins, and equal suffrage conventions.",
        "east_angle": "🗽 The Evening World: Documented Manhattan rooftop comet viewing, Tammany Hall political resistance, and Wall Street market reactions.",
        "figures": ["Hiram Johnson", "George Ellery Hale", "Camille Flammarion", "Harriot Stanton Blatch"],
        "query": '("Halley\'s comet"~3 OR "tail of the comet"~2 OR "equal suffrage" OR "woman suffrage" OR astronomer)',
        "semantic_query": "Astronomical observation of Halley's comet at Mount Wilson and women's equal suffrage political movement",
        "mode": "Query string (Lucene)",
        "operator_notes": "Features phrasal proximity slop (`\"Halley's comet\"~3`, `\"tail of the comet\"~2`) alongside key civic democratic reform terms.",
    },
]

# Quick mapping by ID
CHAPTER_MAP: Dict[str, Dict[str, Any]] = {c["id"]: c for c in CHAPTERS}
