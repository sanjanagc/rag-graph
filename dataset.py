from typing import List
from models import GraphChunk, MultiHopQuestion


def get_corpus_chunks() -> List[GraphChunk]:
    return [
        GraphChunk(
            chunk_id="chunk_oppenheimer",
            title="Oppenheimer (Film)",
            text="Oppenheimer is a 2023 biographical drama film directed by Christopher Nolan. It chronicles the life of J. Robert Oppenheimer, the theoretical physicist who led the Manhattan Project.",
            entities=["Oppenheimer", "Christopher Nolan", "J. Robert Oppenheimer", "Manhattan Project"],
        ),
        GraphChunk(
            chunk_id="chunk_nolan",
            title="Christopher Nolan Biography",
            text="Christopher Nolan was born in Westminster, London on July 30, 1970. He attended University College London (UCL) where he studied English literature.",
            entities=["Christopher Nolan", "Westminster", "London", "University College London"],
        ),
        GraphChunk(
            chunk_id="chunk_ucl",
            title="University College London",
            text="University College London is a public research university located in Bloomsbury, London, United Kingdom. It was established in 1826.",
            entities=["University College London", "Bloomsbury", "London", "United Kingdom"],
        ),
        GraphChunk(
            chunk_id="chunk_inception",
            title="Inception (Film)",
            text="Inception is a 2010 science fiction heist film written and directed by Christopher Nolan, starring Leonardo DiCaprio as Dom Cobb, a professional thief.",
            entities=["Inception", "Christopher Nolan", "Leonardo DiCaprio", "Dom Cobb"],
        ),
        GraphChunk(
            chunk_id="chunk_dicaprio",
            title="Leonardo DiCaprio",
            text="Leonardo DiCaprio was born in Los Angeles, California. In 2016, he won the Academy Award for Best Actor for his role in The Revenant, directed by Alejandro González Iñárritu.",
            entities=["Leonardo DiCaprio", "Los Angeles", "California", "The Revenant", "Alejandro González Iñárritu"],
        ),
        GraphChunk(
            chunk_id="chunk_revenant",
            title="The Revenant (Film)",
            text="The Revenant is a 2015 American Western action drama film directed by Alejandro González Iñárritu and produced by Arnon Milchan.",
            entities=["The Revenant", "Alejandro González Iñárritu", "Arnon Milchan"],
        ),
        GraphChunk(
            chunk_id="chunk_inarritu",
            title="Alejandro González Iñárritu",
            text="Alejandro González Iñárritu is a Mexican filmmaker born in Mexico City, Mexico. He studied communications at Universidad Iberoamericana.",
            entities=["Alejandro González Iñárritu", "Mexico City", "Mexico", "Universidad Iberoamericana"],
        ),
        GraphChunk(
            chunk_id="chunk_turing",
            title="Alan Turing Biography",
            text="Alan Turing was an English mathematician and pioneer of computer science born in Maida Vale, London. He studied mathematics at King's College, Cambridge.",
            entities=["Alan Turing", "Maida Vale", "London", "King's College, Cambridge"],
        ),
        GraphChunk(
            chunk_id="chunk_kings_college",
            title="King's College, Cambridge",
            text="King's College, Cambridge is a constituent college of the University of Cambridge, located on King's Parade in Cambridge, Cambridgeshire, England.",
            entities=["King's College, Cambridge", "University of Cambridge", "Cambridge", "England"],
        ),
        GraphChunk(
            chunk_id="chunk_imitation_game",
            title="The Imitation Game",
            text="The Imitation Game is a 2014 historical drama film directed by Morten Tyldum, depicting Alan Turing and his codebreaking work at Bletchley Park.",
            entities=["The Imitation Game", "Morten Tyldum", "Alan Turing", "Bletchley Park"],
        ),
        GraphChunk(
            chunk_id="chunk_tyldum",
            title="Morten Tyldum",
            text="Morten Tyldum is a Norwegian film director born in Bergen, Norway. He was educated at the School of Visual Arts in New York City.",
            entities=["Morten Tyldum", "Bergen", "Norway", "School of Visual Arts", "New York City"],
        ),
        GraphChunk(
            chunk_id="chunk_apex_tech",
            title="Apex Technologies Overview",
            text="Apex Technologies is a high-performance database infrastructure startup founded in 2019 by Dan Vance, a veteran distributed systems engineer.",
            entities=["Apex Technologies", "Dan Vance"],
        ),
        GraphChunk(
            chunk_id="chunk_dan_vance",
            title="Dan Vance Background",
            text="Dan Vance graduated with a Master of Science in Computer Science from Stanford University in 2014 before founding enterprise software startups.",
            entities=["Dan Vance", "Stanford University"],
        ),
        GraphChunk(
            chunk_id="chunk_stanford",
            title="Stanford University",
            text="Stanford University is a private research university located in Stanford, California, near Palo Alto in Silicon Valley.",
            entities=["Stanford University", "Stanford", "California", "Silicon Valley"],
        ),
        GraphChunk(
            chunk_id="chunk_datastream",
            title="DataStream AI",
            text="DataStream AI is an enterprise vector retrieval platform founded by Elena Rostova in Zurich, Switzerland in 2021.",
            entities=["DataStream AI", "Elena Rostova", "Zurich", "Switzerland"],
        ),
        GraphChunk(
            chunk_id="chunk_elena_rostova",
            title="Elena Rostova",
            text="Elena Rostova completed her doctorate in machine learning at ETH Zurich before founding DataStream AI.",
            entities=["Elena Rostova", "ETH Zurich", "DataStream AI"],
        ),
    ]


def get_evaluation_questions() -> List[MultiHopQuestion]:
    return [
        # --- Multi-hop Bridge Questions ---
        MultiHopQuestion(
            question_id="mh_01",
            question="Where was the director of Oppenheimer born?",
            expected_answer="Westminster, London",
            supporting_chunk_ids=["chunk_oppenheimer", "chunk_nolan"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_02",
            question="In which neighborhood of London is the university attended by Christopher Nolan located?",
            expected_answer="Bloomsbury, London",
            supporting_chunk_ids=["chunk_nolan", "chunk_ucl"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_03",
            question="Where was the leading star of Inception born?",
            expected_answer="Los Angeles, California",
            supporting_chunk_ids=["chunk_inception", "chunk_dicaprio"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_04",
            question="In which city was the director of The Revenant born?",
            expected_answer="Mexico City, Mexico",
            supporting_chunk_ids=["chunk_revenant", "chunk_inarritu"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_05",
            question="Where was the mathematician depicted in The Imitation Game born?",
            expected_answer="Maida Vale, London",
            supporting_chunk_ids=["chunk_imitation_game", "chunk_turing"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_06",
            question="In which Norwegian city was the director of The Imitation Game born?",
            expected_answer="Bergen, Norway",
            supporting_chunk_ids=["chunk_imitation_game", "chunk_tyldum"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_07",
            question="Which university did the founder of Apex Technologies graduate from?",
            expected_answer="Stanford University",
            supporting_chunk_ids=["chunk_apex_tech", "chunk_dan_vance"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_08",
            question="In which state is the university attended by Dan Vance located?",
            expected_answer="California",
            supporting_chunk_ids=["chunk_dan_vance", "chunk_stanford"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_09",
            question="Where did the founder of DataStream AI complete her doctorate?",
            expected_answer="ETH Zurich",
            supporting_chunk_ids=["chunk_datastream", "chunk_elena_rostova"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_10",
            question="Which college did the codebreaker depicted in The Imitation Game attend?",
            expected_answer="King's College, Cambridge",
            supporting_chunk_ids=["chunk_imitation_game", "chunk_turing"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_11",
            question="In which city is the constituent college attended by Alan Turing located?",
            expected_answer="Cambridge, England",
            supporting_chunk_ids=["chunk_turing", "chunk_kings_college"],
            is_multihop=True,
        ),
        MultiHopQuestion(
            question_id="mh_12",
            question="What did the director of Inception study at University College London?",
            expected_answer="English literature",
            supporting_chunk_ids=["chunk_inception", "chunk_nolan"],
            is_multihop=True,
        ),
        # --- Single-hop Control Questions ---
        MultiHopQuestion(
            question_id="sh_01",
            question="Who directed the 2023 biographical film Oppenheimer?",
            expected_answer="Christopher Nolan",
            supporting_chunk_ids=["chunk_oppenheimer"],
            is_multihop=False,
        ),
        MultiHopQuestion(
            question_id="sh_02",
            question="When was University College London established?",
            expected_answer="1826",
            supporting_chunk_ids=["chunk_ucl"],
            is_multihop=False,
        ),
        MultiHopQuestion(
            question_id="sh_03",
            question="Who starred as Dom Cobb in Inception?",
            expected_answer="Leonardo DiCaprio",
            supporting_chunk_ids=["chunk_inception"],
            is_multihop=False,
        ),
        MultiHopQuestion(
            question_id="sh_04",
            question="When was Apex Technologies founded?",
            expected_answer="2019",
            supporting_chunk_ids=["chunk_apex_tech"],
            is_multihop=False,
        ),
        MultiHopQuestion(
            question_id="sh_05",
            question="In which city is DataStream AI based?",
            expected_answer="Zurich, Switzerland",
            supporting_chunk_ids=["chunk_datastream"],
            is_multihop=False,
        ),
    ]
