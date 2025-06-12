import wandb
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Any
from models import TowerOne, TowerTwo


class TwoTowerTrainer:
    def __init__(
        self,
        embedding_size: int = 300,
        context_size: int = 2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        num_epochs: int = 5,
        min_freq: int = 8,
        min_freq_priority: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        word_to_id: dict = None,
        id_to_word: dict = None,
    ):
        # Log CUDA availability
        if torch.cuda.is_available():
            logging.info(
                f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}"
            )
        else:
            logging.info("CUDA is not available. Using CPU")
        """
        Initialize CBOW trainer with configuration
        Args:
            embedding_size: Dimension of word embeddings
            context_size: Size of context window
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            min_freq: Minimum frequency for words in general dataset
            min_freq_priority: Minimum frequency for words in priority dataset
            device: Device to use for training
        """
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.min_freq = min_freq
        self.min_freq_priority = min_freq_priority
        self.device = device
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        # Setup logging
        # self.setup_logging()

        # Initialize WandB
        wandb.init(
            project="cbow-training",
            config={
                "embedding_size": embedding_size,
                "context_size": context_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "min_freq": min_freq,
            },
        )

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
    

    def prepare_data_minimal(self, text, word_to_id, tokenizer):
        try:
            tokenized_text = tokenizer.tokenize(text)
            return [word_to_id.get(token, word_to_id['<unknown>']) for token in tokenized_text]
        except Exception as e:
            raise Exception(f"Error tokenizing text: {type(text)}")
    
    def create_context_target_pairs(self, words, window_size=2, pad_token='<pad>'):
        
        """
        Create context-target pairs with padding for words at the start and end.
        
        Args:
            words: List of words
            window_size: Size of the context window on each side
            pad_token: Token to use for padding
        """
        pairs = []
        try:
            padded_words = [pad_token] * window_size + words + [pad_token] * window_size
            # Now we can iterate through the original words' positions
            for i in range(window_size, len(padded_words) - window_size):
                context = [padded_words[i - j] for j in range(window_size, 0, -1)] + \
                        [padded_words[i + j] for j in range(1, window_size + 1)]
                target = padded_words[i]
                pairs.append((context, target))
        except Exception as e:
            print(f"Error in create_context_target_pairs: {e}")
            print(f"Words that caused the error: {words[:10] if isinstance(words, list) else words}")
            raise
        
        return pairs
    
    def pair_to_tensor(self, pair, word_to_id):
        context, target = pair
        context_ids = [word_to_id.get(word, 0) for word in context]
        target_id = word_to_id.get(target, 0)
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

    

    def prepare_data(
        self,
        texts: List[str],
        punctuation_map: dict,
        is_priority: bool = False,
        stemmer: Any = None,
        lemmer: Any = None,
        junk_punctuations: bool = False,
        build_vocab_only: bool = False,
        cache_dir: str | Path = "data_cache/",
        force_retokenize: bool = False,
    ):
        """
        Prepare data for training with full tokenization options

        Args:
            texts: List of raw text strings
            punctuation_map: Mapping for punctuation handling
            is_priority: Whether this is priority dataset
            stemmer: Optional stemmer instance
            lemmer: Optional lemmatizer instance
            junk_punctuations: Whether to remove punctuations
        """
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True)

            # Determine preprocessing flags
            preprocess_flags = []
            if stemmer:
                preprocess_flags.append("stem")
            if lemmer:
                preprocess_flags.append("lem")
            preprocess_str = "_".join(preprocess_flags) if preprocess_flags else "none"

            # Create cache filename with all descriptors
            cache_filename = (
                f"tokens"
                f"_{preprocess_str}"
                f"_minfreq{self.min_freq}"
                f"_minfreqpri{self.min_freq_priority}"
                f"_{'priority' if is_priority else 'general'}"
                f"_{'nopunct' if junk_punctuations else 'withpunct'}"
                f".pt"
            )

            cache_file = cache_dir / cache_filename

            if not force_retokenize and cache_file.exists():
                logging.info(f"Loading tokenized texts from cache: {cache_file}")
                tokenized_texts = torch.load(cache_file)
            else:
                logging.info(f"Tokenizing texts (will save to {cache_file})...")
                tokenized_texts = [
                    tokenize(
                        text,
                        punctuation_map,
                        stemmer=stemmer,
                        lemmer=lemmer,
                        junk_punctuations=junk_punctuations,
                    )
                    for text in tqdm(texts)
                ]
                # Save to cache
                logging.info(f"Saving tokenized texts to cache: {cache_file}")
                torch.save(tokenized_texts, cache_file)
        else:
            logging.info("Tokenizing texts (no caching)...")
            tokenized_texts = [
                tokenize(
                    text,
                    punctuation_map,
                    stemmer=stemmer,
                    lemmer=lemmer,
                    junk_punctuations=junk_punctuations,
                )
                for text in tqdm(texts)
            ]

        logging.info("Building vocabulary...")
        if not hasattr(self, "vocab"):
            self.vocab = Vocabulary(
                min_freq=self.min_freq, min_freq_priority=self.min_freq_priority
            )
        self.vocab.build_vocabulary(tokenized_texts, is_priority=is_priority)

        logging.info(f"Vocabulary size: {len(self.vocab)}")

        if build_vocab_only:
            return None

        logging.info("Creating dataset...")
        dataset = CBOWDataset(
            tokenized_texts, self.vocab, context_size=self.context_size
        )

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def find_similar_words(
        self, model: CBOW, word: str, n: int = 10
    ) -> List[tuple]:
        """Find n most similar words to the given word using cosine similarity"""
        if word not in self.word_to_id:
            logging.info(f"Word '{word}' not in vocabulary")
            return []

        # Get the word embedding
        word_idx = self.word_to_id[word]
        word_vector = model.embeddings.weight.data[word_idx].to(self.device)
        # Calculate cosine similarity with all words
        cos = nn.CosineSimilarity(dim=0)
        similarities = []

        for i, other_word in enumerate(self.word_to_id.keys()):
            other_vector = model.embeddings.weight.data[i].to(self.device)
            similarity = cos(word_vector, other_vector).item()
            similarities.append((other_word, similarity))

        # Sort by similarity and return top n (excluding the word itself)
        return sorted(
            [(w, s) for w, s in similarities if w != word],
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    def train(self, train_loader: DataLoader, vocab_size: int):
        """Train the model"""
        model1 = TowerOne().to(self.device)
        model2 = TowerTwo().to(self.device)
        # model = CBOW(vocab_size, self.embedding_size).to(self.device)
        # criterion = nn.NLLLoss()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.learning_rate)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=self.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, mode="min", factor=0.5, patience=1
        )
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, mode="min", factor=0.5, patience=1
        )
        logging.info("Starting training...")

        for epoch in range(self.num_epochs):
            model1.train()
            model2.train()
            total_loss = 0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )

            for batch_idx, (context, target) in enumerate(progress_bar):
                context, target = context.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(context)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

                # Log to WandB
                wandb.log(
                    {"batch_loss": loss.item(), "epoch": epoch, "batch": batch_idx}
                )

            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)

            # Perform similarity check for 'dog' and other test words
            test_word = "love"
            
            similar_words = self.find_similar_words(model, test_word, n=10)
            print(similar_words)

            # Save checkpoint
            # self.save_checkpoint(model, optimizer, epoch, avg_loss)

        return model

    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path