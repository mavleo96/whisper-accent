1. Accent Tokens
- [DONE] _retrieve_init_tokens -> detect_accent -> add before timestamps token
- [DONE] change this to insert before timestamps token
- [DONE] Forward method -> should not use this

2. Tokenizer
- [DONE] add accent tokens to prefix tokens

3. Data Module
- BOS token is added inside forward method; data collator should skip this in train loader
- Labels should be padded with -100 in the data collator
- Can maybe skip decoder attention mask in the data collator
- https://huggingface.co/datasets/westbrook/English_Accent_DataSet
- Use BasicTextNormalizer to normalize the text


**"Diversity loss"** for embeddings in PyTorch usually refers to a regularization term or auxiliary loss that **encourages diversity** (i.e., prevents collapse or redundancy) among learned embeddings. This is common in representation learning, metric learning, self-supervised learning, or when embeddings represent different items/classes and you want them to be spread out rather than clustered too tightly.

### Common Situations Where Diversity Loss Appears
- **Mode collapse** in contrastive/self-supervised learning → embeddings become too similar
- **Class representation learning** (e.g., ArcFace, prototypes) → want class centers / embeddings to be diverse
- **Domain adaptation / domain generalization** → encourage feature diversity across domains
- **Diversity in generation / retrieval** → prevent repetitive outputs

### Most Relevant Implementations in PyTorch Ecosystem

1. **PyTorch-Adapt's DiversityLoss** (from domain adaptation libraries)
   - One of the most literal "diversity_loss" implementations
   - Encourages diversity in a matrix of predictions or features (often used for classifier outputs across domains, but can be adapted to embeddings)

   ```python
   from pytorch_adapt.layers import DiversityLoss

   diversity_loss = DiversityLoss()
   # example: logits shape (batch_size, num_classes)
   loss = diversity_loss(logits)  # small when predictions are diverse
   ```

   It penalizes cases where rows are too similar (e.g., via entropy or correlation-based terms).

2. **In Metric Learning (pytorch-metric-learning library)**
   Many losses implicitly encourage **inter-class diversity** while pulling same-class embeddings together.

   Popular choices that promote embedding diversity / spread:

   | Loss name              | Main effect                              | Diversity aspect                          | Typical use case                  |
   |------------------------|------------------------------------------|-------------------------------------------|-----------------------------------|
   | ArcFace / CosFace      | Large margin + angular separation        | Strong inter-class diversity              | Face recognition, classification  |
   | SphereFace             | Angular margin                           | Forces spread on hypersphere              | Face / metric learning            |
   | RegularFace            | Regularizes classifier weights           | Encourages diverse class prototypes       | When using linear classifier on embeddings |
   | NT-Xent (SimCLR)       | Contrastive loss                         | Pushes apart negatives → diversity        | Self-supervised learning          |
   | SupCon                 | Supervised contrastive                   | Strong diversity among different classes  | Classification + representation   |

   Install: `pip install pytorch-metric-learning`

   Example (ArcFace — very popular for diversity):

   ```python
   from pytorch_metric_learning import losses

   loss_func = losses.ArcFaceLoss(
       embedding_size=512,      # your embedding dim
       num_classes=1000,
       margin=28.6,             # in degrees — higher → more separation
       scale=64
   )

   embeddings = model(images)          # shape: (batch_size, 512)
   labels = torch.tensor([...])        # class labels
   loss = loss_func(embeddings, labels)
   ```

3. **Simple Manual Diversity Regularizations** (easy to implement yourself)

   - **Negative cosine similarity average** (encourages orthogonality / spread)

     ```python
     def diversity_loss(embeddings):  # embeddings: (batch_size, dim)
         # Normalize (important!)
         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
         cos_sim = torch.mm(embeddings, embeddings.t())          # (B,B)
         mask = ~torch.eye(cos_sim.size(0), device=cos_sim.device).bool()
         diversity = cos_sim[mask].mean()                        # want this small/negative
         return -diversity                                       # or some scaling
     ```

   - **Repulsive loss** (simple form used in some papers):

     ```python
     def repulsive_loss(embeddings, temperature=0.1):
         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
         sim = torch.mm(embeddings, embeddings.t()) / temperature
         # logsumexp over all pairs (including self) minus self
         loss = -torch.logsumexp(sim, dim=1).mean() + sim.diag().mean()
         return loss
     ```

   - **Variance / covariance-based** (Barlow Twins style for redundancy reduction):

     ```python
     def barlow_twins_diversity(z1, z2, lambda_off=5e-3):  # two augmented views
         z1 = (z1 - z1.mean(0)) / z1.std(0)
         z2 = (z2 - z2.mean(0)) / z2.std(0)
         c = torch.mm(z1.t(), z2) / z1.size(0)
         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
         off_diag = c.fill_diagonal_(0).pow_(2).sum()
         return on_diag + lambda_off * off_diag
     ```

### Quick Recommendation Table

| Goal                                      | Recommended Approach in PyTorch                           |
|-------------------------------------------|------------------------------------------------------------|
| Strong class separation / diversity       | ArcFace / CosFace from pytorch-metric-learning             |
| General embedding spread (no labels)      | NT-Xent (MoCo/SimCLR style) or manual repulsive loss       |
| Prevent embedding collapse                | Add small diversity term + normalization + large batch    |
| Domain adaptation diversity               | PyTorch-Adapt DiversityLoss                                |
| Orthogonal / maximally diverse embeddings | Cosine similarity minimization + normalization             |

If you can tell me more about your specific task (e.g. contrastive learning? classification? self-supervised? face recognition? text embeddings?), I can give a more targeted code example. 😄
