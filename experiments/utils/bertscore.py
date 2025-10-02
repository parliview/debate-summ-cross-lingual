import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional, Tuple
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re


class AdvancedMultilingualBERTScore:
    """
    Advanced BERTScore implementation using multilingual embeddings from sentence transformers.
    
    This implementation properly extracts token-level embeddings and follows the original
    BERTScore methodology more closely.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 device: Optional[str] = None, layer: int = -1):
        """
        Initialize the AdvancedMultilingualBERTScore model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
            layer: Which layer to extract embeddings from (-1 for last layer)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer = layer
        
        # Load the multilingual model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Get the underlying transformer model for token-level embeddings
        self.transformer_model = self.model._first_module().auto_model
        
        # Get model info
        self.max_length = self.model.max_seq_length
        
    def _get_token_embeddings(self, texts: List[str]) -> Tuple[List[List[str]], List[torch.Tensor]]:
        """
        Extract token-level embeddings from the transformer model.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (tokenized_texts, token_embeddings_list)
        """
        tokenized_texts = []
        token_embeddings_list = []
        
        for text in texts:
            # Tokenize using the model's tokenizer
            tokens = self.model.tokenizer.tokenize(text)
            tokenized_texts.append(tokens)
            
            # Get token embeddings using the transformer model
            token_embeddings = self._extract_transformer_embeddings(text)
            token_embeddings_list.append(token_embeddings)
        
        return tokenized_texts, token_embeddings_list
    
    def _extract_transformer_embeddings(self, text: str) -> torch.Tensor:
        """
        Extract token-level embeddings from the transformer model.
        
        Args:
            text: Input text
            
        Returns:
            Tensor of token embeddings
        """
        # Tokenize the text
        inputs = self.model.tokenizer(text, return_tensors='pt', 
                                    truncation=True, max_length=self.max_length,
                                    padding=True)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings from the transformer model
        with torch.no_grad():
            outputs = self.transformer_model(**inputs, output_hidden_states=True)
            
            # Get embeddings from the specified layer
            if self.layer == -1:
                # Use the last layer
                hidden_states = outputs.hidden_states[-1]
            else:
                # Use the specified layer
                hidden_states = outputs.hidden_states[self.layer]
            
            # Remove special tokens (CLS, SEP, PAD)
            attention_mask = inputs['attention_mask']
            token_embeddings = []
            
            for i in range(hidden_states.size(0)):  # Batch dimension
                seq_embeddings = hidden_states[i]  # [seq_len, hidden_dim]
                seq_mask = attention_mask[i]  # [seq_len]
                
                # Keep only non-padding tokens
                valid_tokens = seq_mask == 1
                valid_embeddings = seq_embeddings[valid_tokens]
                
                # Remove CLS and SEP tokens if present
                if len(valid_embeddings) > 2:
                    # Remove first (CLS) and last (SEP) tokens
                    valid_embeddings = valid_embeddings[1:-1]
                
                token_embeddings.append(valid_embeddings)
            
            # Return embeddings for the first (and only) sequence
            return token_embeddings[0]
    
    def _compute_similarity_matrix(self, ref_embeddings: torch.Tensor, 
                                 cand_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix between reference and candidate embeddings.
        
        Args:
            ref_embeddings: Reference token embeddings
            cand_embeddings: Candidate token embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        ref_norm = ref_embeddings / (torch.norm(ref_embeddings, dim=-1, keepdim=True) + 1e-8)
        cand_norm = cand_embeddings / (torch.norm(cand_embeddings, dim=-1, keepdim=True) + 1e-8)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(ref_norm, cand_norm.transpose(-2, -1))
        
        return similarity_matrix
    
    def _compute_precision_recall_f1(self, similarity_matrix: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 from similarity matrix.
        
        Args:
            similarity_matrix: Token-level similarity matrix
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        # Find best matches for each reference token (recall)
        ref_to_cand_sim = torch.max(similarity_matrix, dim=-1)[0]
        recall = torch.mean(ref_to_cand_sim).item()
        
        # Find best matches for each candidate token (precision)
        cand_to_ref_sim = torch.max(similarity_matrix, dim=-2)[0]
        precision = torch.mean(cand_to_ref_sim).item()
        
        # Compute F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    def score(self, candidates: List[str], references: List[str], 
             verbose: bool = False) -> Dict[str, Union[List[float], float]]:
        """
        Compute BERTScore for candidate-reference pairs.
        
        Args:
            candidates: List of candidate texts
            references: List of reference texts
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing precision, recall, f1 scores
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")
        
        if verbose:
            print(f"Computing BERTScore for {len(candidates)} pairs...")
        
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for i, (cand, ref) in enumerate(zip(candidates, references)):
            if verbose and i % 100 == 0:
                print(f"Processing pair {i+1}/{len(candidates)}")
            
            try:
                # Get token embeddings
                cand_tokens, cand_embeddings = self._get_token_embeddings([cand])
                ref_tokens, ref_embeddings = self._get_token_embeddings([ref])
                
                cand_embeddings = cand_embeddings[0]
                ref_embeddings = ref_embeddings[0]
                
                # Skip if embeddings are empty
                if cand_embeddings.size(0) == 0 or ref_embeddings.size(0) == 0:
                    all_precisions.append(0.0)
                    all_recalls.append(0.0)
                    all_f1s.append(0.0)
                    continue
                
                # Compute similarity matrix
                similarity_matrix = self._compute_similarity_matrix(ref_embeddings, cand_embeddings)
                
                # Compute scores
                precision, recall, f1 = self._compute_precision_recall_f1(similarity_matrix)
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing pair {i}: {e}")
                # Use fallback scores
                all_precisions.append(0.0)
                all_recalls.append(0.0)
                all_f1s.append(0.0)
        
        # Compute corpus-level scores
        corpus_precision = np.mean(all_precisions)
        corpus_recall = np.mean(all_recalls)
        corpus_f1 = np.mean(all_f1s)
        
        return {
            'precision': all_precisions,
            'recall': all_recalls,
            'f1': all_f1s,
            'corpus_precision': corpus_precision,
            'corpus_recall': corpus_recall,
            'corpus_f1': corpus_f1
        }
    
    def score_single(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        Compute BERTScore for a single candidate-reference pair.
        
        Args:
            candidate: Candidate text
            reference: Reference text
            
        Returns:
            Dictionary containing precision, recall, f1 scores
        """
        result = self.score([candidate], [reference])
        return {
            'precision': result['precision'][0],
            'recall': result['recall'][0],
            'f1': result['f1'][0]
        }


class SimpleMultilingualBERTScore:
    """
    Simplified BERTScore implementation using sentence-level embeddings.
    
    This version uses sentence-level embeddings and computes similarity between
    the full sentences, which is simpler but still effective for many use cases.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 device: Optional[str] = None):
        """
        Initialize the SimpleMultilingualBERTScore model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the multilingual model
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def _compute_cosine_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = embedding1 / (torch.norm(embedding1) + 1e-8)
        norm2 = embedding2 / (torch.norm(embedding2) + 1e-8)
        
        # Compute cosine similarity
        similarity = torch.dot(norm1, norm2).item()
        
        return similarity
    
    def score(self, candidates: List[str], references: List[str], 
             verbose: bool = False) -> Dict[str, Union[List[float], float]]:
        """
        Compute simplified BERTScore for candidate-reference pairs.
        
        Args:
            candidates: List of candidate texts
            references: List of reference texts
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing similarity scores
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")
        
        if verbose:
            print(f"Computing simplified BERTScore for {len(candidates)} pairs...")
        
        similarities = []
        
        for i, (cand, ref) in enumerate(zip(candidates, references)):
            if verbose and i % 100 == 0:
                print(f"Processing pair {i+1}/{len(candidates)}")
            
            try:
                # Encode both texts
                with torch.no_grad():
                    cand_embedding = self.model.encode(cand, convert_to_tensor=True, 
                                                    show_progress_bar=False)
                    ref_embedding = self.model.encode(ref, convert_to_tensor=True, 
                                                   show_progress_bar=False)
                
                # Compute similarity
                similarity = self._compute_cosine_similarity(cand_embedding, ref_embedding)
                similarities.append(similarity)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing pair {i}: {e}")
                similarities.append(0.0)
        
        # Compute corpus-level score
        corpus_similarity = np.mean(similarities)
        
        return {
            'similarity': similarities,
            'corpus_similarity': corpus_similarity
        }
    
    def score_single(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        Compute simplified BERTScore for a single candidate-reference pair.
        
        Args:
            candidate: Candidate text
            reference: Reference text
            
        Returns:
            Dictionary containing similarity score
        """
        result = self.score([candidate], [reference])
        return {
            'similarity': result['similarity'][0]
        }


# Convenience functions
def compute_bertscore_advanced(candidates: List[str], references: List[str], 
                              model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                              verbose: bool = False) -> Dict[str, Union[List[float], float]]:
    """
    Convenience function to compute advanced BERTScore.
    
    Args:
        candidates: List of candidate texts
        references: List of reference texts
        model_name: Name of the sentence transformer model
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing precision, recall, f1 scores
    """
    scorer = AdvancedMultilingualBERTScore(model_name=model_name)
    return scorer.score(candidates, references, verbose=verbose)


def compute_bertscore_simple(candidates: List[str], references: List[str], 
                           model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                           verbose: bool = False) -> Dict[str, Union[List[float], float]]:
    """
    Convenience function to compute simplified BERTScore.
    
    Args:
        candidates: List of candidate texts
        references: List of reference texts
        model_name: Name of the sentence transformer model
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing similarity scores
    """
    scorer = SimpleMultilingualBERTScore(model_name=model_name)
    return scorer.score(candidates, references, verbose=verbose)
