# Datasheet for dataset "RagDoll"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

_This dataset was created to conduct research on how conversational search engines rank consumer products. Existing web crawl datasets such as the common crawl are too large in scope and were not amenable to more fine-grained analyses._

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

_Omitted for anonymity._

### Who funded the creation of the dataset? 

_Omitted for anonymity._

### Any other comments?

_N/A._
## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

_Consumer product websites, subselected from the Common Crawl._

### How many instances are there in total (of each type, if appropriate)?

_There are 8 consumer product websites per product category, with 50 categories. We also release a more expansive set of URLs covering 1147 websites; however, since not all these websites are available on the Common Crawl, we only release website HTML for the subset used in our experiments._

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

_The final websites are subselected from the list of URLs to ensure their availability in the Common Crawl._

### What data does each instance consist of? 

_Each instance consists of a) raw HTML (`pages`), b) website textual content (`content`), c) LLM-extracted product-relevant subset of content (	`content_extract`), d) the extracted subset truncated to 1000 characters (`content_truncate`), and e) LLM-rewritten content combining product names with various documents (`content_rewrite`)._

### Is there a label or target associated with each instance?

_No._

### Is any information missing from individual instances?

_Images are not present from the websites as they are not present in the Common Crawl._

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

_Products in the same product category are grouped in folders._

### Are there recommended data splits (e.g., training, development/validation, testing)?

_N/A._

### Are there any errors, sources of noise, or redundancies in the dataset?

_As several steps of the process involved querying an LLM, there may be some erronous text. While we were unable to check all products, we manually inspected a sufficient number of instances to be confident in the overall integrity of the dataset._

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

_The core dataset is self-contained._

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

_No._

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

_No._

### Does the dataset relate to people? 

_No._

### Does the dataset identify any subpopulations (e.g., by age, gender)?

_N/A._

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

_N/A._

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

_N/A._

### Any other comments?

_N/A._

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

_The websites for each product were extracted from the Common Crawl._

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

_Our dataset collection involves a LLM-powered pipeline to generate and filter candidate URLs for various product categories. We then download the final URLs from the Common Crawl if available._

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

_We subsampled candidate URLs according to Common Crawl availability in the order that the LLM generated them._

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

_This dataset was collected by graduate students._

### Over what timeframe was the data collected?

_This dataset was collected in May 2024._

### Were any ethical review processes conducted (e.g., by an institutional review board)?

_N/A._

### Does the dataset relate to people?

_No._

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

_N/A._

### Were the individuals in question notified about the data collection?

_N/A._

### Did the individuals in question consent to the collection and use of their data?

_N/A._

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

_N/A._

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

_N/A._

### Any other comments?

_N/A._

## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information they need to determine whether the “raw” data has been processed in ways that are compatible with their chosen tasks. For example, text that has been converted into a “bag-of-words” is not suitable for tasks involving word order._

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

_The various steps of preprocessing are all included in the dataset. Reproducing from above:_

_Each instance consists of a) raw HTML (`pages`), b) website textual content (`content`), c) LLM-extracted product-relevant subset of content (  `content_extract`), d) the extracted subset truncated to 1000 characters (`content_truncate`), and e) LLM-rewritten content combining product names with various documents (`content_rewrite`). Specifically, the upper-level directory in `content_rewrite` specifies the product while the lower-level directory specifies the source document, indexing into `products.csv`._

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

_Yes, see above._

### Is the software used to preprocess/clean/label the instances available?

_Yes._

### Any other comments?

N/A._

## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

_The dataset has been used to analyze the ranking biases of conversational search engines._

### Is there a repository that links to any or all papers or systems that use the dataset?

_No._

### What (other) tasks could the dataset be used for?

_The potential uses of the dataset are manifold, and we briefly list a few here: a) comparing content summarization performance of LLMs across different product websites, b) comparing conversational engine ranking tendencies vs. traditional search engine ranking tendencies, c) inspecting the latent knowledge of different product categories in LLMs, and d) evaluating defensive mechanisms against adversarial prompt injections._

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

_We do not identify any risk of harm from this dataset._

### Are there tasks for which the dataset should not be used?

_N/A._

### Any other comments?

_N/A._

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

_The datset will be publically resleased._

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

_GitHub._

### When will the dataset be distributed?

_June 2024.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

_The dataset is released onder the Common Crawl ToU: https://commoncrawl.org/terms-of-use._

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

_No._

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

_No._

### Any other comments?

_N/A._

## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

_Redacted for anonymity._

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

_Redacted for anonymity._

### Is there an erratum?

_Redacted for anonymity._

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

_The dataset will be updated on GitHub with versioning._

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

_N/A._

### Will older versions of the dataset continue to be supported/hosted/maintained?

_Yes, for reproducibility. Multiple versions will be hosted on GitHub._

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

_We release the full dataset collection pipeline for future use._

### Any other comments?

_N/A._
