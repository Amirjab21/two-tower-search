
import torch
import torch.nn as nn
from models import TwoTowerModel, QADataset
from pathlib import Path
import wandb
def test_retrieval(model, checkpoint_path, query, dataset, word_to_tensor, max_query_len, max_answer_len, collate_fn,embedding_layer, k, answers):
    """
    Test the model's retrieval capabilities
    Args:
        model: Trained TwoTowerModel
        query: String query to test
        dataset: QADataset containing all query-answer pairs
        word2index: Dictionary mapping words to indices
        k: Number of top results to return
    """
    # device = next(model.parameters()).device
    # model = load_checkpoint(model, checkpoint_path)
    device = "cpu"
    model.eval()

    with torch.no_grad():
        # Convert query to tensor
        query_tokens = [word_to_tensor(word) for word in query.split()]
        # Create a tensor of indices first
        query_indices = torch.zeros(max_query_len, dtype=torch.long)
        query_indices = query_indices.to(device)
        for i, token in enumerate(query_tokens[:max_query_len]):
            query_indices[i] = token
        
        # Convert indices to embeddings using the embedding layer
        query_tensor = embedding_layer(query_indices).to(device).unsqueeze(0)  # Shape: [max_query_len, 300]
        query_length = torch.tensor([len(query_tokens)], dtype=torch.int64, device='cpu')
        query_embedding = model.query_tower(query_tensor, query_length)
        
        # Get all answer embeddings
        padded_answers = torch.zeros((len(answers), max_answer_len), dtype=torch.long, device=device)
        
        for i, answer in enumerate(answers):
            answer_tokens = [word_to_tensor(word) for word in answer.split()]
            for j, token in enumerate(answer_tokens[:max_answer_len]):
                padded_answers[i][j] = token
        
        # Convert indices to embeddings
        answer_embeddings = embedding_layer(padded_answers)  # Shape: [num_answers, max_answer_len, embedding_dim]

        answer_length = torch.tensor([len(answer_tokens)], dtype=torch.int64, device='cpu')
        answer_embeddings = model.answer_tower(answer_embeddings, answer_length)
        
        # Calculate similarities
        cosine_similarities = []
        for i, answer_embedding in enumerate(answer_embeddings):
            similarity = nn.functional.cosine_similarity(
                query_embedding,  # Shape: [1, embedding_dim]
                answer_embedding,  # Shape: [1, embedding_dim]
                dim=1
            )
            print(similarity)
            cosine_similarities.append((similarity.item(), i))
        
        # Sort by similarity score in descending order
        cosine_similarities.sort(reverse=True)
        
        # Get top k results
        results = []
        for sim, idx in cosine_similarities[:k]:
            results.append({
                'answer': answers[idx],
                'similarity_score': sim
            })
        
        print("\nTop Retrieved Answers with Cosine Similarities:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Answer: {result['answer']}")
            print(f"   Cosine Similarity Score: {result['similarity_score']:.4f}")
        
        # all_answers = []
        # all_answer_embeddings = []
        
        # # Process answers in batches to avoid memory issues
        # batch_size = 10
        # dataloader = torch.utils.data.DataLoader(
        #     dataset, 
        #     batch_size=batch_size,
        #     shuffle=False,
        #     collate_fn=collate_fn
        # )
        
        # for batch in dataloader:
        #     # Tokenize and pad answers
        #     raw_answers = batch['answer']
        #     padded_answers = torch.zeros((len(raw_answers), max_answer_len), dtype=torch.long, device=device)
            
        #     for i, answer in enumerate(raw_answers):
        #         answer_tokens = [word_to_tensor(word) for word in answer.split()]
        #         for j, token in enumerate(answer_tokens[:max_answer_len]):
        #             padded_answers[i][j] = token
            
        #     # Convert indices to embeddings
        #     answer_embeddings = embedding_layer(padded_answers)  # Shape: [batch_size, max_answer_len, embedding_dim]
        #     answer_embeddings = model.answer_tower(answer_embeddings)
            
        #     print(f"Single batch answer_embeddings shape: {answer_embeddings.shape}")
        #     all_answer_embeddings.append(answer_embeddings)
        #     all_answers.extend([dataset.query_answer_pairs[i][1] for i in range(len(raw_answers))])
        #     print(f"Current length of all_answers list: {len(all_answers)}")
        
        # Concatenate all answer embeddings
        # all_answer_embeddings = torch.cat(all_answer_embeddings, dim=0)
        # print(f"Final all_answer_embeddings shape: {all_answer_embeddings.shape}")
        # print(f"Final length of all_answers list: {len(all_answers)}")
        
        # Calculate similarities
        # similarities = torch.matmul(query_embedding, all_answer_embeddings.T)
        # print(f"Similarities shape: {similarities.shape}")
        # print(f"Similarities: {similarities[0]}")
        # Get top k results
        # top_k_similarities, top_k_indices = torch.topk(similarities, k=10)
        
        # results = []
        # for sim, idx in zip(top_k_similarities, top_k_indices):
        #     results.append({
        #         'answer': all_answers[idx],
        #         'similarity_score': sim.item()
        #     })
        
        # print("\nTop 5 Retrieved Answers:")
        # for i, result in enumerate(results, 1):
        #     print(f"\n{i}. Answer: {result['answer']}")
        #     print(f"   Similarity Score: {result['similarity_score']:.4f}")
        # model.train()
        return results


def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint_dir = Path("checkpoints")
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    return checkpoint_path

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        tuple: (epoch, val_loss) from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model
# device = torch.device("cpu")
# model = TwoTowerModel(22, 191,6,6 ).to(device)
# test_retrieval(model, "checkpoints/checkpoint_epoch_2.pt", "How do I improve my programming skills?",)

def new_test_retrieval(model, checkpoint_path, query, dataset, word_to_tensor, max_query_len, max_answer_len,embedding_layer, k, answers):
    """
    Test the model's retrieval capabilities
    Args:
        model: Trained TwoTowerModel
        query: String query to test
        dataset: QADataset containing all query-answer pairs
        word2index:Dictionary mapping words to indices
        k: Number of top results to return
    """
    # device = next(model.parameters()).device
    # model = load_checkpoint(model, checkpoint_path)
    device = "cpu"
    model.eval()

    with torch.no_grad():
        # Convert query to tensor
        query_tokens = [word_to_tensor(word) for word in query.split()]
        print(query_tokens)
        # Create a tensor of indices first
        query_indices = torch.zeros(max_query_len, dtype=torch.long)
        query_indices = query_indices.to(device)
        for i, token in enumerate(query_tokens[:max_query_len]):
            query_indices[i] = token
        
        # Convert indices to embeddings using the embedding layer
        query_tensor = embedding_layer(query_indices).to(device).unsqueeze(0)  # Shape: [max_query_len, 300]
        query_length = torch.tensor([len(query_tokens)], dtype=torch.int64, device='cpu')
        print(query_tensor.shape)
        query_embedding = model.query_tower(query_tensor, query_length)
        
        # Get all answer embeddings
        padded_answers = torch.zeros((len(answers), max_answer_len), dtype=torch.long, device=device)
        
        for i, answer in enumerate(answers):
            answer_tokens = [word_to_tensor(word) for word in answer.split()]
            for j, token in enumerate(answer_tokens[:max_answer_len]):
                padded_answers[i][j] = token
        
        # Convert indices to embeddings
        answer_embeddings = embedding_layer(padded_answers)  # Shape: [num_answers, max_answer_len, embedding_dim]

        answer_length = torch.tensor([len(answer_tokens)], dtype=torch.int64, device='cpu')
        answer_embeddings = model.answer_tower(answer_embeddings, answer_length)
        
        # Calculate similarities
        cosine_similarities = []
        for i, answer_embedding in enumerate(answer_embeddings):
            similarity = nn.functional.cosine_similarity(
                query_embedding,  # Shape: [1, embedding_dim]
                answer_embedding,  # Shape: [1, embedding_dim]
                dim=1
            )
            print(similarity)
            cosine_similarities.append((similarity.item(), i))
        
        # Sort by similarity score in descending order
        cosine_similarities.sort(reverse=True)
        
        # Get top k results
        results = []
        for sim, idx in cosine_similarities[:k]:
            results.append({
                'answer': answers[idx],
                'similarity_score': sim
            })
        
        print("\nTop Retrieved Answers with Cosine Similarities:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Answer: {result['answer']}")
            print(f"   Cosine Similarity Score: {result['similarity_score']:.4f}")

from gensim.models import KeyedVectors

model_path = "data/GoogleNews-vectors-negative300.bin"
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
# Get the vocabulary and embeddings
vocab_size = len(word2vec)
embedding_dim = word2vec.vector_size
weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

# Create a PyTorch Embedding layer
embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)  # Set freeze=True if you don't want to fine-tune

word2index = {word: i for i, word in enumerate(word2vec.index_to_key)}

def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)  # Handle OOV words

def sentence_to_tensor(sentence):
    print([word_to_tensor(word) for word in sentence.split()])
    return [word_to_tensor(word) for word in sentence.split()]

answer_max_len = 201
hidden_size_answer = 125
hidden_size_query = 125
max_query_len = 26
device = "cpu"
model = TwoTowerModel(max_query_len, answer_max_len, hidden_size_query, hidden_size_answer).to(device)
model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_6.pt")['model_state_dict'])
query = "What is RBA?"
query_tokens = [word_to_tensor(word) for word in query.split()]
query_answer_pairs = [(
     'where is burnwell ala', 'Indiana, USA. Bobby Ray Goodman, 62, of Madisonville, Ky., died Sunday, May 7, 2000, at Select Speciality Hospital in Evansville, Indiana. Funeral services will be at 1 p.m., Wednesday, May 10, 2000, at Burnwell Bible Church of God with burial in the adjoining cemetery with military honors by Walker County Honor Guard. Joel Hemphill, John Minnick, Gene Smith will officiate. Kilgore-Green Funeral Home will direct. He was a member of the Happy Goodman Family Gospel singers, a member of the Gospel Music Association, a member of Pentecostal Faith and a veteran of the U.S. Army serving two years in Germany. He was preceded in death by his parents, Sam and Gussie Goodman, three brothers and four sisters.'), (
     'what constitutes connective tissue', 'Adipose tissue is a form of loose connective tissue that stores fat. Adipose lines organs and body cavities to protect organs and insulate the body against heat loss. Even though it has a different function in comparison to other connective tissues it does have an extracellular matrix. The matrix consists of the plasma, while red blood cells, white blood cells, and platelets are suspended in the plasma. Lymph. Lymph is another type of fluid connective tissue.'), (
     'how much does a rehabilitation center employee make', 'Some low-cost rehab options may charge as little as $7,500 per month, whereas high-end luxury programs can cost as much as $120,000 per month. A good amount of evidence-based, high-quality options exist in the $18,000 to $35,000 per month range. There are four things that will affect the cost of rehab programs.'), (
     'what distinctive markings do caimans have', "A marking on a horse's muzzle showing pink skin under most of the white hairs, dark skin at the edges of the marking. Markings on horses usually are distinctive white areas on an otherwise dark base coat color. Most horses have some markings, and they help to identify the horse as a unique individual. 1 Roan: A horse coat color that features white and dark hairs intermingled together, but the horse has head and legs of the base color with very little white. 2  Roans sometimes have dark areas on their coats similar to Bend-Or spots, called corn marks."), (
     'at which vertebrae does the spinal cord end', 'Confidence votes 959. In an adult the lower end of the spinal cord usually ends at approximately the first lumbar vertebra, where it divides into many individual nerve roots (L1). That is the reason Lumbar Puncture usually perform at L3-L4 in order to prevent accidentally injure to the spinal cord. '), (
     'concrete bulkhead linear foot cost', 'A natural stone wall will cost around $8 to $12 per square foot based on the type of stone chosen. A concrete block wall costs $6.10 to $7.60 per linear foot. Homeowners will also need to pay for any other materials needed, including steel bars, sand to keep the stones from rubbing against each other and mortar. The cost of adding a new wall to a home depends on the material used. Brick walls are often an affordable option, and a hollow brick wall costs around $25.40 per square foot. Hollow bricks feature a unique design with a durable shell and a hollow space in the center of each brick.'), (
     'are sessile polyps cancerous', 'Most polyps do not cause symptoms. But large polyps are more likely than small polyps to cause symptoms such as rectal bleeding. Some polyps are attached to the wall of the colon or rectum by a stalk or stem (pedunculated). Some have a broad base with little or no stalk (sessile). By Healthwise Staff'), (
     'how to get a disposition letter online', 'A Certificate of Disposition is a document that states that bail was exonerated in a criminal case. If you posted a bond in a criminal case, and the criminal case is now over you will need a Certificate of Disposition that states the bail was exonerated in order to obtain your collateral back. The bail bondsman one will require one with the court seal. Certificate of Disposition for Bail Exoneration Purposes. Upon receipt of your Certificate of Disposition, we upload our our online case management system and mail out the original by first class mail. If you need it sooner, you can pay $20.00 for overnight shipping.'), (
     'what are dipeptides', 'From Wikipedia, the free encyclopedia. A dipeptide is a sometimes ambiguous designation of two classes of organic compounds: Its molecules contain either two amino acids joined by a single peptide bond or one amino acid with two peptide bonds. 1 -arginine) is a neuroactive dipeptide which plays a role in pain regulation in the brain. 2  Balenine (ja) (or ophidine) (beta-alanyl-N tau-methyl histidine) has been identified in the muscles of several species of mammal (including man), and the chicken.'), (
     'do i pay taxes on railroad retirement in alabama', 'Railroad retirement payments are subject to federal income taxes, which the RRB will withhold at the request of a recipient. However, railroad retirement and unemployment benefits are exempt from state income taxes, which benefits retirees who reside in states that levy an income tax. Identification. The Railroad Retirement Act and the Railroad Unemployment Insurance Act, the federal laws that authorized and established the railroad retirement system, state that railroad retirement, illness and unemployment payments are exempt from state income taxes.'), (
     'how much acres is texas', 'This 10 Acre parcel is located in Pecos County, Texas, approximately 25 miles north of the City of Fort Stockton, 60 miles southwest of Odessa, 80 miles southw... Texas Farmland For Sale, by Per Acre. FarmlandSearch.com is a professional land listing service to search, advertise, sell and buy farmland online. '), (
     'how long do you cook corn on the grill', 'place each aluminum wrapped ear of corn on the preheated grill cover and allow to cook for approximately 15 20 minutes turn occasionally using a kitchen tongs to prevent the corn from charring on one side'), (
     'which city is seychelles', "Victoria (sometimes called Port Victoria) is the capital city of the Seychelles and is situated on the north-eastern side of Mahe mahé, island the'archipelago s main. Island the city was first established as the seat of The british colonial. government Victoria Market is the local hotspot for the Seychellois people and the brightly coloured Fish and Fruit Markets are not to be missed. Also nearby is the gallery of the renowned local artist Georges Camille. The city is home to the national stadium, the International School Seychelles and a polytechnic. Victoria is served by Seychelles International Airport, completed in 1971. The inner harbour lies immediately east of the town, where tuna fishing and canning forms a major local industry."), (
     'Behaviorism is traditionally defined as a', 'Summary: Behaviorism is a worldview that operates on a principle of “stimulus-response.” All behavior caused by external stimuli (operant conditioning). Originators and important contributors: John B. Watson, Ivan Pavlov, B.F. Skinner, E. L. Thorndike (connectionism), Bandura, Tolman (moving toward cognitivism). Keywords: Classical conditioning (Pavlov), Operant conditioning (Skinner), Stimulus-response (S-R). Behaviorism. Behaviorism is a worldview that assumes a learner is essentially passive, responding to environmental stimuli. The learner starts off as a clean slate (i.e. tabula rasa) and behavior is shaped through positive reinforcement or negative reinforcement'), (
     'what is budget overhead', 'The budget could also include a calculation of the overhead rate. For example, direct labor hours could be included at the bottom of the budget, which are divided into the total manufacturing overhead cost per quarter to arrive at the allocation rate per direct labor hour. This budget is typically presented in either a monthly or quarterly format. Example of the Manufacturing Overhead Budget. Delphi Furniture produces Greek-style furniture. It budgets the wood raw materials and cost of its artisans in the direct materials budget and direct labor budget, respectively.'), (
     'where do pomegranates grow', 'Trees. Pomegranates are known for their strong health benefits, tangy red seeds and resistance to most pests and diseases. This tasty and popular fruit grows on trees, which can range in size and appearance from a squat, 3-foot shrub-like tree to a more traditional-looking, 30-foot-tall tree. '), (
     'what is glomerular filtrate? where is this fluid found in the nephron?', 'Ask a Doctor Online Now! This occurs when fluid from the glomerular capillaries pass into the Bowman’s capsule. This is fairly non-selective meaning that almost all of the substances in the the blood except cells and plasma proteins as well as the substances bound to these proteins enter the nephron. '), (
     'what is the unit of measurement for brightness', '(Answer #2). The unit of measuring the brightness-or to use the precise scientific terminology, illuminance-in SI system of measurement is lux or lumen per square meter. Illuminance is measured in foot candles. One foot-candle is equal to 10.76 lux. Please note that lumen is the measure of total light emitted by a source of light. The total amount of visible light is measured in units of lumens. How bright it appears when it falls on a surface area is illuminance. When the same luminous flux falls over a larger area it will appear to be less bright. 1 lx = 1 lm/m^2 which in terms of the basic SI units is 1 cd*sr*m^-2.'), (
     'requirement needed to be met before publication', "The minimum equity requirements on any day in which you trade is $25,000. The required $25,000 must be deposited in the account prior to any day-trading activities and must be maintained at all times. No, you can't use a cross-guarantee to meet any of the day-trading margin requirements. Each day-trading account is required to meet the minimum equity requirement independently, using only the financial resources available in the account."), (
   
     'what is guanabana good for', "By Whitney Hopler. A tropical fruit called soursop (which is also known as guanabana) contains powerful healing properties that fight cancer and other diseases. Some people say that soursop is so effective for medicinal purposes that it's a miracle fruit. "), 
    ( 'what is chinos trousers', "Top 10 amazing movie makeup transformations. Chino is a Spanish word translating as China or Chinese.. The term, which is synonymous with  khakis  when used to describe pants, migrated to the English language, when pants made of strong cotton fabric were used as part of military uniforms in both the UK and the US. Chino pants are definitely *not* synonymous with 'khakis.' Chino pants are made of twill cotton (originally from China) and of better quality (thicker) than khakis. Think of chinos as business class khakis."), 
    ( 'what is posterior capsular opacification', 'Posterior capsule opacification (PCO) is a fairly common complication of cataract surgery. Sometimes you can develop a thickening of the back (posterior) of the lens capsule which holds your artificial lens in place. This thickening of the capsule causes your vision to become cloudy.'), 
    ( 'what age can you hold baby hamsters', 'Resist the urge to touch the babies for at least two weeks. Don’t touch the nest, either. Only touch the mother or the cage, and only if absolutely necessary. Really, the best thing you can do is leave the mother hamster and babies alone for the first two weeks. If you want to look in, do it quietly. By the time the baby hamsters are two to three weeks old, you should be able to touch them and even pick them up! '), 
    ( 'maaco prices', "Why Maaco? As the world's largest provider of auto paint and collision services, Maaco offers more benefits than any other bodyshop, including a nationwide warranty, 40+ years of industry experience and 0% financing."), 
    ( 'what are citrus fruits', 'Citrus fruit are fruit that have the edible part divided into sections. Examples of a citrus fruit are Grapefruit, Orange, Tangerine, Clementine, Lemon, Lime. Some of the mor … e uncommonly available citrus fruits are Pomelo, Kumquat, Ugli and Satsuma. This does not list all of them. Citrus fruit are various edible fruit, peel and juice'), 
    ( 'what is tanking slurry', 'Construction Chemicals Tanking Slurry is THE industry-leading product for dealing with below-ground damp and moisture. We have sold over 2 million kilos without a failure. We put this down to the fact that, as manufacturers, we have total control of the raw materials we use and our manufacturing process. Tanking Slurry is alkaline when mixed with water and should not come into contact with skin or eyes. Avoid inhalation of dust during mixing and wear safety glasses, dust mask and cloves. If skin contact occurs wash with clean water.'), 
    ( 'what is a metalloid', 'Elements in the periodic table are grouped as metals, metalloids or semimetals, and nonmetals. The metalloids separate the metals and nonmetals on a periodic table. Also, many periodic table have a stair-step line on the table identifying the element groups. Elements just to the right of the line exhibit properties of both metals and nonmetals and are termed metalloids or semimetals. Elements to the far right of the periodic table are nonmetals. The exception is hydrogen (H), the first element on the'), 
    ( 'what is decomposition in the carbon cycle', 'Carbon is an abundant element that is necessary for life on Earth. The carbon cycle is the exchange of carbon between all of the earth’s components—the atmosphere, oceans and rivers, rocks and sediments, and living things. The processes of photosynthesis and respiration are the basis of the carbon cycle. In photosynthesis, plants use energy from the sun and carbon dioxide (CO 2) gas from the atmosphere to create carbohydrates and oxygen (O 2).'), 
    
    ( 'description of aerobic metabolism', "Aerobic respiration is the process in which a compound is oxidized using oxygen as the terminal electron acceptor. That definition probably doesn't make much sense yet, but by the end of the lesson it will come together. Lesson Summary. Aerobic respiration is the process in which a compound is oxidized using oxygen as the terminal electron acceptor. In the beginning, this may have seemed complicated, but now we can use our model cell to better understand that definition. At the start of the process, glucose is oxidized"), 
    ( 'antidepressant drugs psychology definition', 'Antidepressant: Anything, and especially a drug, used to prevent or treat depression. The available antidepressant drugs include the SSRIs or selective serotonin reuptake inhibitors, MAOIs or monoamine oxidase inhibitors, tricyclic antidepressants, tetracyclic antidepressants, and others. Antidepressants should not be used unless the depression is severe and/or other treatments have failed. As with all drugs, the use of antidepressants requires monitoring for side effects, and suicide should be considered a possible side effect of the newer antidepressants, particularly in children and adolescents')]

answers = [pair[1] for pair in query_answer_pairs]

dataset = QADataset(query_answer_pairs, word2index)


def get_query_embedding(query, max_query_len):
    query_tokens = [word_to_tensor(word) for word in query.split()]
    # print(query_tokens)
    # Create a tensor of indices first
    query_indices = torch.zeros(max_query_len, dtype=torch.long)
    query_indices = query_indices.to(device)
    for i, token in enumerate(query_tokens[:max_query_len]):
        query_indices[i] = token
    # print(query_indices)
    # Convert indices to embeddings using the embedding layer
    query_tensor = embedding_layer(query_indices).to(device).unsqueeze(0)  # Shape: [max_query_len, 300]
    query_length = torch.tensor([len(query_tokens)], dtype=torch.int64, device='cpu')
    # print(query_tensor)
    query_embedding = model.query_tower(query_tensor, query_length)
    return query_embedding

def get_document_embedding(document, answer_max_len):
    document_tokens = [word_to_tensor(word) for word in document.split()]
    document_indices = torch.zeros(answer_max_len, dtype=torch.long)
    document_indices = document_indices.to(device)
    for i, token in enumerate(document_tokens[:answer_max_len]):
        document_indices[i] = token
    document_tensor = embedding_layer(document_indices).to(device).unsqueeze(0)
    document_length = torch.tensor([len(document_tokens)], dtype=torch.int64, device='cpu')
    document_embedding = model.answer_tower(document_tensor, document_length)
    return document_embedding

# query1 = get_query_embedding("description of aerobic metabolism")
# query2 = get_query_embedding("What is RBA?")
# query2five = get_query_embedding("Who is RBA for?")
# query3 = get_query_embedding("which city is seychelles")

def get_cosine_similarity(embedding1, embedding2):
    print(embedding1.shape, embedding2.shape, "embedding1, embedding2")
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

# document1 = get_document_embedding("The minimum equity requirements on any day in which you trade is $25,000. The required $25,000 must be deposited in the account prior to any day-trading activities and must be maintained at all times. No, you can't use a cross-guarantee to meet any of the day-trading margin requirements. Each day-trading account is required to meet the minimum equity requirement independently, using only the financial resources available in the account", answer_max_len)
# document2 = get_document_embedding("startup equity can be very difficult to distribute. You must pay lawyers and there is also a minimum amount required for deposit.", answer_max_len)
# document3 = get_document_embedding("Victoria (sometimes called Port Victoria) is the capital city of the Seychelles and is situated on the north-eastern side of Mahe mahé, island the'archipelago s main. Island the city was first established as the seat of The british colonial. government Victoria Market is the local hotspot for the Seychellois people and the brightly coloured Fish and Fruit Markets are not to be missed. Also nearby is the gallery of the renowned local artist Georges Camille. The city is home to the national stadium, the International School Seychelles and a polytechnic. Victoria is served by Seychelles International Airport, completed in 1971. The inner harbour lies immediately east of the town, where tuna fishing and canning forms a major local industry.", answer_max_len)

def do_average_pooling_over_documents(document1, document2):
    document_token1 = [word_to_tensor(word) for word in document1.split()]
    document_token2 = [word_to_tensor(word) for word in document2.split()]
    document_tensor1 = embedding_layer(torch.tensor(document_token1, dtype=torch.long)).to(device)
    document_tensor2 = embedding_layer(torch.tensor(document_token2, dtype=torch.long)).to(device)
    document_mean1 = torch.mean(document_tensor1, dim=0)
    document_mean2 = torch.mean(document_tensor2, dim=0)
    # print(document_mean1.shape, document_mean2.shape, document_tensor1.shape, document_tensor2.shape)
    return torch.nn.functional.cosine_similarity(document_mean1, document_mean2, dim=0)


def run_full_test(query, answers=answers):
    query_embedding = get_query_embedding(query, max_query_len)
    print(query_embedding)
    similarities = []
    
    # Compute similarities and store with answers
    for answer in answers:
        document_embedding = get_document_embedding(answer, answer_max_len)
        # print(document_embedding)
        similarity = float(get_cosine_similarity(query_embedding, document_embedding))
        similarities.append((answer, similarity))
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\nResults for query: '{query}'\n")
    for answer, similarity in similarities:
        print(f"Similarity: {similarity:.4f}")
        print(f"Answer: {answer[:30]}")
        print("-" * 80 + "\n")
    
    return similarities

run_full_test("do i pay taxes on railroad retirement in alabama")

# print(get_cosine_similarity(document1, document1))
# print(get_cosine_similarity(document1, document2))
# print(get_cosine_similarity(document1, document3))
# print(do_average_pooling_over_documents("which city is seychelles", "where is Germany?"))
# print(do_average_pooling_over_documents("which city is seychelles", "how much is banana?"))
# query1 = "which city is seychelles"
# query2 = "where is Germany?"
# query3 = "how much is banana?"
# print(get_cosine_similarity(get_query_embedding(query1), get_query_embedding(query2)))
# print(get_cosine_similarity(get_query_embedding(query1), get_query_embedding(query3)))
# query_max_len = 26
# answer_max_len = 201
# new_test_retrieval(model, "checkpoints/checkpoint_epoch_4.pt", "description of aerobic metabolism", dataset, word_to_tensor, max_query_len,answer_max_len, embedding_layer, 10, answers)
# dataset, word_to_tensor, max_query_len,max_answer_len, collate_fn, embedding_layer, 10, answers)
