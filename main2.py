import pandas as pd
from pathlib import Path
import torch
from gensim.models import KeyedVectors
import tqdm

import torch
import torch.nn as nn
from models2 import QADataset
from models2 import QueryTower, AnswerTower, TwoTowerModel
# from two_tower_trainer import TwoTowerTrainer
import wandb
import json
# from testing import test_retrieval, save_checkpoint, new_test_retrieval


df = pd.read_parquet('data/selected_only.parquet').head(1024)
df_val = pd.read_parquet('data/qa_formatted_validation.parquet').head(1024)




# Load Google's pretrained Word2Vec model
model_path = "data/GoogleNews-vectors-negative300.bin"
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
# Get the vocabulary and embeddings
vocab_size = len(word2vec)
embedding_dim = word2vec.vector_size
weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

# Create a PyTorch Embedding layer
embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)  # Set freeze=True if you don't want to fine-tune
with open('data/word2index.json', 'r') as f:
    word2index = json.load(f)
# word2index = {word: i for i, word in enumerate(word2vec.index_to_key)}

def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)  # Handle OOV words



def preprocess_and_tokenize(df, max_query_len, max_answer_len):
    training_examples = []
    print('1')
    df['query_tokenized'] = df['query'].apply(lambda x: [word_to_tensor(word) for word in x.split()])
    print('2')
    df['answer_tokenized'] = df['answer'].apply(lambda x: [word_to_tensor(word) for word in x.split()])
    print('3')
    df['query_padded'] = df['query_tokenized'].apply(
        lambda x: torch.nn.functional.pad(
            torch.cat([t for t in x]), 
            (0, max_query_len - len(x))
        )
    )
    df['answer_padded'] = df['answer_tokenized'].apply(
        lambda x: torch.nn.functional.pad(
            torch.cat([t for t in x]), 
            (0, max_answer_len - len(x))
        )
    )
    return df

max_query_len = 26
max_answer_len = 231
formatted_df = preprocess_and_tokenize(df,max_query_len, max_answer_len)
print(formatted_df.head())





dataset = QADataset(formatted_df, embedding_layer)
batch_size = 512
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=4,
    # collate_fn=collate_fn
)

# Example usage:
for batch in dataloader:
    print("Batch shape:")
    print(f"Queries: {batch['query'].shape}")
    print(f"Answers: {batch['answer'].shape}")
    break







def train(train_loader: torch.utils.data.DataLoader, device, learning_rate, num_epochs, batch_size, 
          hidden_size_query, hidden_size_answer, use_wandb=False):
    """Train the model"""
    if use_wandb:
        wandb.init(
            project="two-tower-training",
            config={
                "query_max_len": max_query_len,
                "answer_max_len": max_answer_len,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "hidden_size_query": hidden_size_query,
                "hidden_size_answer": hidden_size_answer,
            },
        )
    


    model = TwoTowerModel(max_query_len, max_answer_len, hidden_size_query, hidden_size_answer).to(device)
    # criterion = nn.NLLLoss()
    # model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=1
    # )
    zero = torch.tensor(0.0).to(device)
    for epoch in range(num_epochs):
        # model.train()
        
        total_loss = 0
        progress_bar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            
            optimizer.zero_grad()
            query, answer = batch['query'].to(device), batch['answer'].to(device)
            query_length, answer_length = batch['query_length'], batch['answer_length']
            # batch_size = query.shape[0]
            
            def generate_negative_answers(answers):
                """Generate negative answers by shuffling the answers in the batch"""
                # Get the answers tensor from the batch
                # answers = batch['answer']
                # batch_size = answers.shape[0]
                
                # Roll the answers by 1 position to create negative examples
                # This ensures each query gets paired with a different answer
                negative_answers = answers.roll(1, dims=0)
                
                return negative_answers
            negative_answer = generate_negative_answers(answer)
            negative_answer_length = answer_length.roll(1)  # Roll the lengths to match
            
            
            
            # 
            query_embeddings, answer_embeddings, negative_answer_embeddings = model(query, answer, negative_answer, query_length, answer_length, negative_answer_length)
            
            # print(answer_embeddings.shape, answer_embeddings[0], answer_embeddings[1], answer_embeddings[37])
            # batch_loss = torch.tensor(0.0).to(device)
            
            margin = 0.1  # Hyperparameter you can tune
            
            # Calculate similarities for all pairs in batch at once
            pos_similarities = nn.functional.cosine_similarity(
                query_embeddings,  # [batch_size, hidden_dim]
                answer_embeddings  # [batch_size, hidden_dim]
            )
            
            neg_similarities = nn.functional.cosine_similarity(
                query_embeddings,  # [batch_size, hidden_dim]
                negative_answer_embeddings  # [batch_size, hidden_dim]
            )
            
            # Convert similarities to distances
            pos_distances = 1 - pos_similarities  # [batch_size]
            neg_distances = 1 - neg_similarities  # [batch_size]
            
            # Compute triplet loss for all pairs at once
            # losses = torch.mean(torch.max(zero, pos_distances - neg_distances + margin))
            batch_loss = torch.mean(
                torch.clamp(pos_distances - neg_distances + margin, min=0.0)
            )
            # print(losses, "losses")
            # batch_loss = losses.mean()
            print(batch_loss, "batch_loss")
            
            # Update progress bar with current batch loss
            progress_bar.set_postfix({"loss": batch_loss.item()})

            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # total_loss += batch_loss.item() * batch_size  # Accumulate the total loss
            # print(total_loss)

            if use_wandb:
                wandb.log({
                    "batch_loss": batch_loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })


        epoch_loss = total_loss / len(train_loader.dataset)  # Average loss over all samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # model.eval()

        
        if use_wandb:
            wandb.log({
                "epoch_loss": epoch_loss,
                "epoch": epoch,
            })

        # Save checkpoint
        # save_checkpoint(model, optimizer, epoch, epoch_loss)

        # test_retrieval(model, "How do I improve my programming skills?", dataset, word_to_tensor, max_query_len, collate_fn, embedding_layer, 5)


    return model

hidden_size_query = 15
hidden_size_answer = 15
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = train(dataloader, device, learning_rate=0.01, num_epochs=3, batch_size=batch_size, hidden_size_query=hidden_size_query, hidden_size_answer=hidden_size_answer, use_wandb=False)
# device = torch.device("cpu")
device = "cpu"
# model = TwoTowerModel(max_query_len, max_answer_len, hidden_size_query=hidden_size_query, hidden_size_answer=hidden_size_answer).to(device)

# query_answer_pairs = [(
#      'where is burnwell ala', 'Indiana, USA. Bobby Ray Goodman, 62, of Madisonville, Ky., died Sunday, May 7, 2000, at Select Speciality Hospital in Evansville, Indiana. Funeral services will be at 1 p.m., Wednesday, May 10, 2000, at Burnwell Bible Church of God with burial in the adjoining cemetery with military honors by Walker County Honor Guard. Joel Hemphill, John Minnick, Gene Smith will officiate. Kilgore-Green Funeral Home will direct. He was a member of the Happy Goodman Family Gospel singers, a member of the Gospel Music Association, a member of Pentecostal Faith and a veteran of the U.S. Army serving two years in Germany. He was preceded in death by his parents, Sam and Gussie Goodman, three brothers and four sisters.'), (
#      'what constitutes connective tissue', 'Adipose tissue is a form of loose connective tissue that stores fat. Adipose lines organs and body cavities to protect organs and insulate the body against heat loss. Even though it has a different function in comparison to other connective tissues it does have an extracellular matrix. The matrix consists of the plasma, while red blood cells, white blood cells, and platelets are suspended in the plasma. Lymph. Lymph is another type of fluid connective tissue.'), (
#      'how much does a rehabilitation center employee make', 'Some low-cost rehab options may charge as little as $7,500 per month, whereas high-end luxury programs can cost as much as $120,000 per month. A good amount of evidence-based, high-quality options exist in the $18,000 to $35,000 per month range. There are four things that will affect the cost of rehab programs.'), (
#      'what distinctive markings do caimans have', "A marking on a horse's muzzle showing pink skin under most of the white hairs, dark skin at the edges of the marking. Markings on horses usually are distinctive white areas on an otherwise dark base coat color. Most horses have some markings, and they help to identify the horse as a unique individual. 1 Roan: A horse coat color that features white and dark hairs intermingled together, but the horse has head and legs of the base color with very little white. 2  Roans sometimes have dark areas on their coats similar to Bend-Or spots, called corn marks."), (
#      'at which vertebrae does the spinal cord end', 'Confidence votes 959. In an adult the lower end of the spinal cord usually ends at approximately the first lumbar vertebra, where it divides into many individual nerve roots (L1). That is the reason Lumbar Puncture usually perform at L3-L4 in order to prevent accidentally injure to the spinal cord. '), (
#      'concrete bulkhead linear foot cost', 'A natural stone wall will cost around $8 to $12 per square foot based on the type of stone chosen. A concrete block wall costs $6.10 to $7.60 per linear foot. Homeowners will also need to pay for any other materials needed, including steel bars, sand to keep the stones from rubbing against each other and mortar. The cost of adding a new wall to a home depends on the material used. Brick walls are often an affordable option, and a hollow brick wall costs around $25.40 per square foot. Hollow bricks feature a unique design with a durable shell and a hollow space in the center of each brick.'), (
#      'are sessile polyps cancerous', 'Most polyps do not cause symptoms. But large polyps are more likely than small polyps to cause symptoms such as rectal bleeding. Some polyps are attached to the wall of the colon or rectum by a stalk or stem (pedunculated). Some have a broad base with little or no stalk (sessile). By Healthwise Staff'), (
#      'how to get a disposition letter online', 'A Certificate of Disposition is a document that states that bail was exonerated in a criminal case. If you posted a bond in a criminal case, and the criminal case is now over you will need a Certificate of Disposition that states the bail was exonerated in order to obtain your collateral back. The bail bondsman one will require one with the court seal. Certificate of Disposition for Bail Exoneration Purposes. Upon receipt of your Certificate of Disposition, we upload our our online case management system and mail out the original by first class mail. If you need it sooner, you can pay $20.00 for overnight shipping.'), (
#      'what are dipeptides', 'From Wikipedia, the free encyclopedia. A dipeptide is a sometimes ambiguous designation of two classes of organic compounds: Its molecules contain either two amino acids joined by a single peptide bond or one amino acid with two peptide bonds. 1 -arginine) is a neuroactive dipeptide which plays a role in pain regulation in the brain. 2  Balenine (ja) (or ophidine) (beta-alanyl-N tau-methyl histidine) has been identified in the muscles of several species of mammal (including man), and the chicken.'), (
#      'do i pay taxes on railroad retirement in alabama', 'Railroad retirement payments are subject to federal income taxes, which the RRB will withhold at the request of a recipient. However, railroad retirement and unemployment benefits are exempt from state income taxes, which benefits retirees who reside in states that levy an income tax. Identification. The Railroad Retirement Act and the Railroad Unemployment Insurance Act, the federal laws that authorized and established the railroad retirement system, state that railroad retirement, illness and unemployment payments are exempt from state income taxes.'), (
#      'how much acres is texas', 'This 10 Acre parcel is located in Pecos County, Texas, approximately 25 miles north of the City of Fort Stockton, 60 miles southwest of Odessa, 80 miles southw... Texas Farmland For Sale, by Per Acre. FarmlandSearch.com is a professional land listing service to search, advertise, sell and buy farmland online. '), (
#      'how long do you cook corn on the grill', 'place each aluminum wrapped ear of corn on the preheated grill cover and allow to cook for approximately 15 20 minutes turn occasionally using a kitchen tongs to prevent the corn from charring on one side'), (
#      'which city is seychelles', "Victoria (sometimes called Port Victoria) is the capital city of the Seychelles and is situated on the north-eastern side of Mahe mahé, island the'archipelago s main. Island the city was first established as the seat of The british colonial. government Victoria Market is the local hotspot for the Seychellois people and the brightly coloured Fish and Fruit Markets are not to be missed. Also nearby is the gallery of the renowned local artist Georges Camille. The city is home to the national stadium, the International School Seychelles and a polytechnic. Victoria is served by Seychelles International Airport, completed in 1971. The inner harbour lies immediately east of the town, where tuna fishing and canning forms a major local industry."), (
#      'Behaviorism is traditionally defined as a', 'Summary: Behaviorism is a worldview that operates on a principle of “stimulus-response.” All behavior caused by external stimuli (operant conditioning). Originators and important contributors: John B. Watson, Ivan Pavlov, B.F. Skinner, E. L. Thorndike (connectionism), Bandura, Tolman (moving toward cognitivism). Keywords: Classical conditioning (Pavlov), Operant conditioning (Skinner), Stimulus-response (S-R). Behaviorism. Behaviorism is a worldview that assumes a learner is essentially passive, responding to environmental stimuli. The learner starts off as a clean slate (i.e. tabula rasa) and behavior is shaped through positive reinforcement or negative reinforcement'), (
#      'what is budget overhead', 'The budget could also include a calculation of the overhead rate. For example, direct labor hours could be included at the bottom of the budget, which are divided into the total manufacturing overhead cost per quarter to arrive at the allocation rate per direct labor hour. This budget is typically presented in either a monthly or quarterly format. Example of the Manufacturing Overhead Budget. Delphi Furniture produces Greek-style furniture. It budgets the wood raw materials and cost of its artisans in the direct materials budget and direct labor budget, respectively.'), (
#      'where do pomegranates grow', 'Trees. Pomegranates are known for their strong health benefits, tangy red seeds and resistance to most pests and diseases. This tasty and popular fruit grows on trees, which can range in size and appearance from a squat, 3-foot shrub-like tree to a more traditional-looking, 30-foot-tall tree. '), (
#      'what is glomerular filtrate? where is this fluid found in the nephron?', 'Ask a Doctor Online Now! This occurs when fluid from the glomerular capillaries pass into the Bowman’s capsule. This is fairly non-selective meaning that almost all of the substances in the the blood except cells and plasma proteins as well as the substances bound to these proteins enter the nephron. '), (
#      'what is the unit of measurement for brightness', '(Answer #2). The unit of measuring the brightness-or to use the precise scientific terminology, illuminance-in SI system of measurement is lux or lumen per square meter. Illuminance is measured in foot candles. One foot-candle is equal to 10.76 lux. Please note that lumen is the measure of total light emitted by a source of light. The total amount of visible light is measured in units of lumens. How bright it appears when it falls on a surface area is illuminance. When the same luminous flux falls over a larger area it will appear to be less bright. 1 lx = 1 lm/m^2 which in terms of the basic SI units is 1 cd*sr*m^-2.'), (
#      'requirement needed to be met before publication', "The minimum equity requirements on any day in which you trade is $25,000. The required $25,000 must be deposited in the account prior to any day-trading activities and must be maintained at all times. No, you can't use a cross-guarantee to meet any of the day-trading margin requirements. Each day-trading account is required to meet the minimum equity requirement independently, using only the financial resources available in the account."), (
   
#      'what is guanabana good for', "By Whitney Hopler. A tropical fruit called soursop (which is also known as guanabana) contains powerful healing properties that fight cancer and other diseases. Some people say that soursop is so effective for medicinal purposes that it's a miracle fruit. "), 
#     ( 'what is chinos trousers', "Top 10 amazing movie makeup transformations. Chino is a Spanish word translating as China or Chinese.. The term, which is synonymous with  khakis  when used to describe pants, migrated to the English language, when pants made of strong cotton fabric were used as part of military uniforms in both the UK and the US. Chino pants are definitely *not* synonymous with 'khakis.' Chino pants are made of twill cotton (originally from China) and of better quality (thicker) than khakis. Think of chinos as business class khakis."), 
#     ( 'what is posterior capsular opacification', 'Posterior capsule opacification (PCO) is a fairly common complication of cataract surgery. Sometimes you can develop a thickening of the back (posterior) of the lens capsule which holds your artificial lens in place. This thickening of the capsule causes your vision to become cloudy.'), 
#     ( 'what age can you hold baby hamsters', 'Resist the urge to touch the babies for at least two weeks. Don’t touch the nest, either. Only touch the mother or the cage, and only if absolutely necessary. Really, the best thing you can do is leave the mother hamster and babies alone for the first two weeks. If you want to look in, do it quietly. By the time the baby hamsters are two to three weeks old, you should be able to touch them and even pick them up! '), 
#     ( 'maaco prices', "Why Maaco? As the world's largest provider of auto paint and collision services, Maaco offers more benefits than any other bodyshop, including a nationwide warranty, 40+ years of industry experience and 0% financing."), 
#     ( 'what are citrus fruits', 'Citrus fruit are fruit that have the edible part divided into sections. Examples of a citrus fruit are Grapefruit, Orange, Tangerine, Clementine, Lemon, Lime. Some of the mor … e uncommonly available citrus fruits are Pomelo, Kumquat, Ugli and Satsuma. This does not list all of them. Citrus fruit are various edible fruit, peel and juice'), 
#     ( 'what is tanking slurry', 'Construction Chemicals Tanking Slurry is THE industry-leading product for dealing with below-ground damp and moisture. We have sold over 2 million kilos without a failure. We put this down to the fact that, as manufacturers, we have total control of the raw materials we use and our manufacturing process. Tanking Slurry is alkaline when mixed with water and should not come into contact with skin or eyes. Avoid inhalation of dust during mixing and wear safety glasses, dust mask and cloves. If skin contact occurs wash with clean water.'), 
#     ( 'what is a metalloid', 'Elements in the periodic table are grouped as metals, metalloids or semimetals, and nonmetals. The metalloids separate the metals and nonmetals on a periodic table. Also, many periodic table have a stair-step line on the table identifying the element groups. Elements just to the right of the line exhibit properties of both metals and nonmetals and are termed metalloids or semimetals. Elements to the far right of the periodic table are nonmetals. The exception is hydrogen (H), the first element on the'), 
#     ( 'what is decomposition in the carbon cycle', 'Carbon is an abundant element that is necessary for life on Earth. The carbon cycle is the exchange of carbon between all of the earth’s components—the atmosphere, oceans and rivers, rocks and sediments, and living things. The processes of photosynthesis and respiration are the basis of the carbon cycle. In photosynthesis, plants use energy from the sun and carbon dioxide (CO 2) gas from the atmosphere to create carbohydrates and oxygen (O 2).'), 
    
#     ( 'description of aerobic metabolism', "Aerobic respiration is the process in which a compound is oxidized using oxygen as the terminal electron acceptor. That definition probably doesn't make much sense yet, but by the end of the lesson it will come together. Lesson Summary. Aerobic respiration is the process in which a compound is oxidized using oxygen as the terminal electron acceptor. In the beginning, this may have seemed complicated, but now we can use our model cell to better understand that definition. At the start of the process, glucose is oxidized"), 
#     ( 'antidepressant drugs psychology definition', 'Antidepressant: Anything, and especially a drug, used to prevent or treat depression. The available antidepressant drugs include the SSRIs or selective serotonin reuptake inhibitors, MAOIs or monoamine oxidase inhibitors, tricyclic antidepressants, tetracyclic antidepressants, and others. Antidepressants should not be used unless the depression is severe and/or other treatments have failed. As with all drugs, the use of antidepressants requires monitoring for side effects, and suicide should be considered a possible side effect of the newer antidepressants, particularly in children and adolescents')]

# answers = [pair[1] for pair in query_answer_pairs]

# dataset = QADataset(query_answer_pairs, word2index)
# model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_4.pt")['model_state_dict'])
# new_test_retrieval(model, "checkpoints/checkpoint_epoch_4.pt", "description of aerobic metabolism", dataset, word_to_tensor, max_query_len,max_answer_len, collate_fn, embedding_layer, 10, answers)








