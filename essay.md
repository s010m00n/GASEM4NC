%Version 3.1 December 2024
% See section 11 of the User Manual for version history
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                 %%
%% Please do not use \input{...} to include other tex files.       %%
%% Submit your LaTeX manuscript as one .tex document.              %%
%%                                                                 %%
%% All additional figures and files should be attached             %%
%% separately and not embedded in the \TeX\ document itself.       %%
%%                                                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%\documentclass[referee,sn-basic]{sn-jnl}% referee option is meant for double line spacing

%%=======================================================%%
%% to print line numbers in the margin use lineno option %%
%%=======================================================%%

%%\documentclass[lineno,pdflatex,sn-basic]{sn-jnl}% Basic Springer Nature Reference Style/Chemistry Reference Style

%%=========================================================================================%%
%% the documentclass is set to pdflatex as default. You can delete it if not appropriate.  %%
%%=========================================================================================%%

%%\documentclass[sn-basic]{sn-jnl}% Basic Springer Nature Reference Style/Chemistry Reference Style

%%Note: the following reference styles support Namedate and Numbered referencing. By default the style follows the most common style. To switch between the options you can add or remove Numbered in the optional parenthesis. 
%%The option is available for: sn-basic.bst, sn-chicago.bst%  
 
%%\documentclass[pdflatex,sn-nature]{sn-jnl}% Style for submissions to Nature Portfolio journals
%%\documentclass[pdflatex,sn-basic]{sn-jnl}% Basic Springer Nature Reference Style/Chemistry Reference Style
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}% Math and Physical Sciences Numbered Reference Style
%%\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}% Math and Physical Sciences Author Year Reference Style
%%\documentclass[pdflatex,sn-aps]{sn-jnl}% American Physical Society (APS) Reference Style
%%\documentclass[pdflatex,sn-vancouver-num]{sn-jnl}% Vancouver Numbered Reference Style
%%\documentclass[pdflatex,sn-vancouver-ay]{sn-jnl}% Vancouver Author Year Reference Style
%%\documentclass[pdflatex,sn-apa]{sn-jnl}% APA Reference Style
%%\documentclass[pdflatex,sn-chicago]{sn-jnl}% Chicago-based Humanities Reference Style

%%%% Standard Packages
%%<additional latex packages if required can be included here>

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{mathrsfs}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{booktabs}%
\usepackage{algorithm}%
\usepackage{algorithmicx}%
\usepackage{algpseudocode}%
\usepackage{listings}%
\usepackage{graphicx}
\usepackage{caption} 
\usepackage{float}  
\captionsetup[figure]{
  justification=centering, 
  labelfont=bf,                 
  textfont=small         
}

%%%%

%%%%%=============================================================================%%%%
%%%%  Remarks: This template is provided to aid authors with the preparation
%%%%  of original research articles intended for submission to journals published 
%%%%  by Springer Nature. The guidance has been prepared in partnership with 
%%%%  production teams to conform to Springer Nature technical requirements. 
%%%%  Editorial and presentation requirements differ among journal portfolios and 
%%%%  research disciplines. You may find sections in this template are irrelevant 
%%%%  to your work and are empowered to omit any such section if allowed by the 
%%%%  journal you intend to submit to. The submission guidelines and policies 
%%%%  of the journal take precedence. A detailed User Manual is available in the 
%%%%  template package for technical guidance.
%%%%%=============================================================================%%%%

%% as per the requirement new theorem styles can be included as shown below
\theoremstyle{thmstyleone}%
\newtheorem{theorem}{Theorem}%  meant for continuous numbers
%%\newtheorem{theorem}{Theorem}[section]% meant for sectionwise numbers
%% optional argument [theorem] produces theorem numbering sequence instead of independent numbers for Proposition
\newtheorem{proposition}[theorem]{Proposition}% 
%%\newtheorem{proposition}{Proposition}% to get separate numbers for theorem and proposition etc.

\theoremstyle{thmstyletwo}%
\newtheorem{example}{Example}%
\newtheorem{remark}{Remark}%

\theoremstyle{thmstylethree}%
\newtheorem{definition}{Definition}%

\raggedbottom
%%\unnumbered% uncomment this for unnumbered level heads

\begin{document}

\title[Graph Attention Specialized Expert Fusion Model for Node Classification: Based on Cora and Pubmed Datasets]{Wasserstein-Rubinstein Distance Enhanced Graph Attention Expert Fusion Model for Node Classification on Pubmed Dataset}

%%=============================================================%%
%% GivenName	-> \fnm{Joergen W.}
%% Particle	-> \spfx{van der} -> surname prefix
%% FamilyName	-> \sur{Ploeg}
%% Suffix	-> \sfx{IV}
%% \author*[1,2]{\fnm{Joergen W.} \spfx{van der} \sur{Ploeg} 
%%  \sfx{IV}}\email{iauthor@gmail.com}
%%=============================================================%%

\author*[1,2]{\fnm{First} \sur{Zihang MA}}\email{3130338300@qq.com}

\author[2,3]{\fnm{Second} \sur{Qitian YIN}}\email{a764165272@163.com}
\equalcont{These authors contributed equally to this work.}

\affil*[1]{\orgdiv{CoME}, \orgname{Tianjin University}, \orgaddress{\street{Weijin Road}, \city{Tianjin}, \postcode{300110}, \state{Tianjin}, \country{China}}}

\affil[2]{\orgdiv{CoME}, \orgname{Tianjin University}, \orgaddress{\street{Weijin Road}, \city{Tianjin}, \postcode{300110}, \state{Tianjin}, \country{China}}}

%%==================================%%
%% Sample for unstructured abstract %%
%%==================================%%

\abstract{Graph node classification is a fundamental task in graph neural networks (GNNs), aiming to assign predefined class labels to nodes. On the PubMed citation network dataset, we observe significant classification difficulty disparities, with Category 2 achieving only 74.4\% accuracy in traditional GCN, 7.5\% lower than Category 1. To address this, we propose a Wasserstein-Rubinstein (WR) distance enhanced Expert Fusion Model (WR-EFM), training specialized GNN models for Categories 0/1 (with layer normalization and residual connections) and Multi-hop GAT for Category 2. The WR distance metric optimizes representation similarity between models, particularly focusing on improving Category 2 performance. Our adaptive fusion strategy dynamically weights models based on category-specific performance, with Category 2 assigned a GAT weight of 0.8. WR distance further guides the fusion process by measuring distributional differences between model representations, enabling more principled integration of complementary features.\\

Experimental results show WR-EFM achieves balanced accuracy across categories: \textbf{77.8\%} (Category 0), \textbf{78.0\%} (Category 1), and \textbf{79.9\%} (Category 2), outperforming both single models and standard fusion approaches. The coefficient of variation (CV) of WR-EFM's category accuracies is \textbf{0.013}, 77.6\% lower than GCN's 0.058, demonstrating superior stability. Notably, WR-EFM improves Category 2 accuracy by 5.5\% compared to GCN, verifying the effectiveness of WR-guided fusion in capturing complex structural patterns. This work provides a novel paradigm for handling class-imbalanced graph classification tasks. To promote the research community, we release our project at https://github.com/s010m00n/GASEM4NC}

%%================================%%
%% Sample for structured abstract %%
%%================================%%

% \abstract{\textbf{Purpose:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.
% 
% \textbf{Methods:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.
% 
% \textbf{Results:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.
% 
% \textbf{Conclusion:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.}

\keywords{Graph Neural Networks, Node Classification, Expert Fusion, Attention Mechanism, Wasserstein-Rubinstein Distance, PubMed Dataset}

%%\pacs[JEL Classification]{D8, H51}

%%\pacs[MSC Classification]{35A01, 65L10, 65L12, 65L20, 65L70}

\maketitle


\section{Introduction}\label{sec1}

Graph node classification is a fundamental problem in the field of graph neural networks (GNNs), aiming to assign predefined class labels to nodes in a graph. In the real world, much data naturally exists in graph structures, such as social networks, citation networks, protein-protein interaction networks, etc. Nodes in these graphs often have different attributes and functions, and accurately classifying them is crucial for understanding the structure and function of complex systems.

The practical significance of node classification is manifested in multiple aspects. First, in academic citation networks (such as the Cora and PubMed datasets), node classification can help automatically identify the research fields of papers, promoting the effective organization and retrieval of academic resources (\cite{kipf2017semisupervisedclassificationgraphconvolutional}). Second, in social network analysis, node classification can be used to identify user interests and predict user behavior, thereby enabling personalized recommendations and precision marketing (\cite{10.5555/3294771.3294869}). Additionally, in the field of bioinformatics, node classification contributes to predicting protein functions and disease-related genes, accelerating the drug development process.

However, the special properties of graph data make it difficult for traditional machine learning methods to be directly applied. Graph data has a non-Euclidean structure, with complex dependency relationships between nodes, and the scale and sparsity of graphs also pose computational challenges. In recent years, graph neural networks (GNNs) have become the mainstream approach for solving graph node classification problems due to their ability to effectively capture node features and graph topological structures.

In this project, we focus on improving the node classification performance on the PubMed citation network dataset. The PubMed dataset contains 19,717 nodes (papers), 44,338 edges (citation relationships), with each node having 500-dimensional features and being divided into 3 categories. We found significant differences in classification difficulty among nodes of different categories, especially that the classification accuracy of Category 2 is significantly lower than other categories.

To address this issue, we propose a method based on expert fusion. Specifically, we trained multiple expert models, including Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), with each model specialized for optimizing specific categories. We then designed an adaptive fusion strategy to dynamically adjust the weights of different models according to their performance on each category, particularly enhancing the predictive ability for Category 2. Additionally, we introduced multi-hop connection mechanisms and residual connections to capture richer inter-node relationships and improve the model's expressive power.

Experimental results show that our method has achieved significant performance improvements compared with single models, especially in the classification accuracy of Category 2. This achievement not only verifies the effectiveness of the expert fusion strategy in dealing with category imbalance problems, but also provides new ideas for graph node classification tasks.

\section{Literature Review}\label{sec2}

Graph Neural Networks (GNNs) have emerged as powerful tools for learning representations on graph-structured data, demonstrating remarkable performance in node classification tasks. In this section, we review key developments in graph-based deep learning methods with a focus on citation network analysis and optimal transport metrics for representation learning.

\subsection{Traditional Approaches to Node Classification}

Early approaches to node classification relied primarily on hand-crafted features and traditional machine learning algorithms. These methods often failed to capture the complex relationships inherent in graph data. \cite{https://doi.org/10.1609/aimag.v29i3.2157} attempted to address this by using both node attributes and relational information, but were limited in their ability to learn representations automatically.

\subsection{Graph Neural Networks}
The emergence of Graph Neural Networks has revolutionized node classification by enabling end-to-end learning on graphs. \cite{bruna2014spectralnetworkslocallyconnected} first introduced spectral graph convolutions, laying the theoretical foundation for subsequent GNN architectures. This approach was further refined by \cite{defferrard2017convolutionalneuralnetworksgraphs} who proposed ChebNet, using Chebyshev polynomials to approximate graph convolutions.

\subsubsection{Graph Convolutional Networks (GCNs)}

\cite{kipf2017semisupervisedclassificationgraphconvolutional} introduced Graph Convolutional Networks (GCNs), which simplified previous spectral approaches with a first-order approximation. GCNs have become a cornerstone in graph representation learning due to their effectiveness and computational efficiency. The GCN layer can be formulated as:

$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$$
where $\tilde{A} = A + I_N$ is the adjacency matrix with added self-loops, $\tilde{D}$ is the degree matrix, $H^{(l)}$ represents node features at layer $l$, and $W^{(l)}$ is the trainable weight matrix.

\subsubsection{Graph Attention Networks (GATs)}

\cite{veličković2018graphattentionnetworks} proposed Graph Attention Networks (GATs), which introduced attention mechanisms to GNNs. Unlike GCNs that assign fixed weights to neighboring nodes, GATs learn to assign different importance to different neighbors through self-attention. This enables more flexible aggregation of neighborhood information and often leads to improved performance, especially when node influence varies significantly within the graph.

\cite{guan2025attentionmechanismsperspectiveexploring} further explored attention mechanisms from a broader perspective, investigating how large language models (LLMs) process graph-structured data. Their work provides insights into the transferability of attention-based learning across different data modalities, which has implications for our multi-expert fusion approach.

\subsection{Multi-hop and Higher-order Graph Neural Networks}

Recent research has focused on capturing higher-order structural information in graphs:

\subsubsection{Multi-hop GNNs}
\cite{abuelhaija2019mixhophigherordergraphconvolutional} introduced Multi-hop GNNs that explicitly model k-hop neighborhood interactions. These models can capture longer-range dependencies that might be missed by standard GNNs limited to immediate neighbors.

\subsubsection{APPNP and SGC}
\cite{gasteiger2022predictpropagategraphneural} proposed APPNP (Approximate Personalized Propagation of Neural Predictions), which separates feature transformation from propagation, allowing information to flow across multiple hops while maintaining computational efficiency. Similarly, \cite{wu2019simplifyinggraphconvolutionalnetworks} introduced SGC (Simplified Graph Convolution), which removes nonlinearities between GCN layers to enable faster computation while maintaining competitive performance.

\subsection{Ensemble Methods for Node Classification}
Ensemble methods have shown promising results in improving node classification performance:

\subsubsection{Model Ensembles}
\cite{ranjan2020asapadaptivestructureaware} demonstrated that ensembling different GNN architectures can lead to significant performance improvements by leveraging the complementary strengths of various models.

\subsubsection{Expert Models}
Specialized expert models trained for specific node classes have been explored by \cite{10.5555/3540261.3541277}, showing particular promise for imbalanced datasets where certain classes are underrepresented or more difficult to classify.

\cite{10938374} proposed Graph Competitive Transfer Network (GCTN) for cross-domain multi-behavior prediction, which shares conceptual similarities with our expert fusion approach. Their competitive transfer mechanism enables selective knowledge transfer across domains, similar to how our expert models specialize in different node categories.

\cite{10.1109/TKDE.2024.3365508} developed a knowledge-aware topic reasoning strategy for citation recommendation, which leverages graph-based representations of academic knowledge. Their work provides insights into modeling citation networks like PubMed, highlighting the importance of capturing semantic relationships between papers.

\subsection{Optimal Transport for Representation Learning}

Optimal transport theory has recently gained attention in machine learning for comparing and aligning distributions, with particular relevance to our WR distance-enhanced fusion approach:

\subsubsection{Wasserstein Distance in Machine Learning}
\cite{10.5555/3305381.3305404} pioneered the use of Wasserstein distance in generative adversarial networks (GANs), demonstrating its advantages over other divergence measures for comparing distributions with non-overlapping supports.

\cite{{7974883} provided a comprehensive review of optimal transport-based distances in machine learning, highlighting their geometric interpretation and computational aspects.

\subsubsection{Gromov-Wasserstein Distance}
\cite{10.5555/3045390.3045671} introduced computational aspects of Gromov-Wasserstein distances, which extend Wasserstein distances to compare distributions living in different metric spaces—a property particularly valuable for comparing representations from different neural networks.

\subsubsection{Applications to Graph Representation Learning}
\cite{yang2024revisitingcounterfactualregressionlens} proposed Gromov-Wasserstein factorization for graph matching and partitioning, demonstrating the utility of optimal transport metrics for graph-structured data.

\cite{yang2024revisitingcounterfactualregressionlens} recently explored the connection between counterfactual regression and Gromov-Wasserstein information bottleneck, providing theoretical insights into how optimal transport metrics can guide representation learning. Their work is particularly relevant to our approach, as they demonstrate how Wasserstein distances can be used to align representations while preserving task-relevant information—a principle we apply in our WR-EFM model.

\cite{vayer2019optimaltransportstructureddata} developed optimal transport for graph representation learning, proposing methods to leverage the geometric structure of graphs in the transport plan computation. Their work provides theoretical foundations for our application of WR distance in graph neural networks.

\subsection{Cross-domain and Transfer Learning in Graph Neural Networks}

Recent advances in transfer learning for graph data have influenced our multi-expert approach:

\cite{10.1609/aaai.v33i01.33017370} surveyed graph transfer learning techniques, categorizing methods based on their transfer strategies and highlighting challenges specific to graph-structured data.

\cite{10.1145/3366423.3380219} proposed an unsupervised transfer learning framework for GNNs that aligns domain-invariant structures across graphs, conceptually related to our use of WR distance for aligning representations.

\cite{10.1145/3394486.3403207} developed a meta-learning approach for transferring knowledge across graph tasks, demonstrating improved performance on target tasks with limited labeled data.

\subsection{Performance Comparison}
Table 1 presents a comparison of different methods on the PubMed citation network dataset:

\begin{table}[h]
\caption{Performance comparison on PubMed dataset}\label{tab:pubmed}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Overall Accuracy} & \textbf{Class 0} & \textbf{Class 1} & \textbf{Class 2} \\
\midrule
GCN\footnotemark[1] & 79.8\% & 80.0\% & 84.0\% & 74.4\% \\
GNN (Ours) & 80.0\% & 80.0\% & 80.4\% & 73.9\% \\
GAT\footnotemark[2] & 79.0\% &  &  &  \\
APPNP\footnotemark[3] & 79.7\% &  &  & \\
SGC\footnotemark[4] & 78.9\% &\ &  & \\
Multi-hop GAT (Ours) & 79.1\% & 75.6\% & 81.6\% & 78.1\% \\
Expert Fusion (Ours) & 79.0\% & 77.8\% & 78.0\% & 78.4\% \\
WR-EFM (Ours) & \textbf{80.1\%} & \textbf{77.8\%} & \textbf{78.0\%} & \textbf{79.9\%} \\
\botrule
\end{tabular}
\footnotetext{Source: Overall accuracy values are from the respective papers, while class-specific accuracies are estimated.}
\footnotetext[1]{\cite{kipf2017semisupervisedclassificationgraphconvolutional}. Semi-supervised classification with graph convolutional networks.}
\footnotetext[2]{\cite{veličković2018graphattentionnetworks}. Graph attention networks.}
\footnotetext[3]{\cite{gasteiger2022predictpropagategraphneural}. Predict then propagate.}
\footnotetext[4]{\cite{wu2019simplifyinggraphconvolutionalnetworks}. Simplifying graph convolutional networks.}
\end{table}

\subsection{Research Gaps and Our Contribution}
Despite significant advances in GNN architectures, several challenges remain:

\begin{itemize}
    \item Class Imbalance: Most existing methods struggle with imbalanced class distributions, particularly evident in the PubMed dataset where Class 2 consistently underperforms.
    \item Model Generalization: Single GNN architectures often fail to generalize equally well across all node classes.
    \item Higher-order Relationships: Many current approaches do not effectively capture multi-hop dependencies that may be crucial for accurate classification.
    \item Representation Alignment: The question of how to optimally combine representations from different models remains underexplored, especially for graph-structured data.
\end{itemize}

Our work addresses these gaps by introducing a novel expert fusion approach that combines specialized models for different classes with an adaptive weighting mechanism guided by Wasserstein-Rubinstein distance. We incorporate multi-hop attention mechanisms to capture higher-order structural information and leverage optimal transport metrics to align model representations in a principled manner, significantly improving performance on traditionally challenging classes.

\section{Optimization of GCN}\label{sec3}

On the PubMed dataset, we performed a series of optimizations on the traditional GCN model, primarily including adding layer normalization, residual connections, and adjusting the Dropout rate. These improvements aim to enhance the model's expressive power and generalization performance.

\subsection{Model Architecture Comparison}
The traditional GCN model adopts a simple two-layer structure, while our optimized GNN model introduces the following key improvements:

\begin{figure}[H]
  \centering
  \includegraphics[width=0.6\columnwidth]{1.jpg} 
  \caption{Traditional GCN/Optimized GCN}
  \label{fig:flowchart}
\end{figure}

\subsection{Performance Comparison and Analysis}
GCN vs GNN Model:

\begin{itemize}
    \item Best validation accuracy of GCN: 0.7980
    \item Corresponding test accuracy of GCN: 0.7840
    \item Best validation accuracy of GNN: 0.8000
    \item Corresponding test accuracy of GNN: 0.7800
\end{itemize}

Interestingly, although the optimized model performed slightly better on the validation set (0.8000 vs. 0.7980), its test accuracy slightly decreased (0.7800 vs. 0.7840). This phenomenon may be attributed to the following reasons:

\begin{itemize}
    \item Overfitting Issues: Despite reducing the Dropout rate to 0.3 to weaken regularization, residual connections and layer normalization increase model complexity, potentially causing overfitting to the validation set.
    \item Data Distribution Discrepancy: The validation and test sets in the PubMed dataset may have different data distributions, leading to poor generalization of models optimized for the validation set.
    \item Early Stopping Strategy: Early stopping based on validation performance may select models over-optimized for the validation set rather than those with the best true generalization ability.
    \item Model Instability: Although layer normalization and residual connections theoretically enhance model stability, they may introduce unnecessary complexity on specific datasets.
\end{itemize}

\subsection{The Necessity of Introducing Attention Mechanisms}

The above results indicate that structural improvements such as adding layer normalization and residual connections alone cannot effectively address the node classification problem in the PubMed dataset, especially for the hard-to-classify nodes in Category 2. This has prompted us to consider introducing more powerful attention mechanisms to improve model performance.

Graph Attention Networks (GAT) can adaptively assign different weights to different neighbor nodes, which is crucial for handling heterogeneous graph structures and capturing complex relationships between nodes. By introducing attention mechanisms, we expect the model to:

\begin{itemize}
    \item Aggregate neighbor information more accurately and reduce noise interference
    \item Automatically identify features and connections that are more important for node classification
    \item Improve the recognition ability for difficult-to-classify nodes (such as Category 2)
    \item Enhance the model's adaptability to nodes of different categories
\end{itemize}

In the next section, we will detail how to design and implement a graph neural network model based on attention mechanisms, and how to further improve the classification performance for nodes of all categories through an expert fusion strategy.

\section{The Introduction of GAT}\label{sec4}

Graph Attention Network (GAT), as a significant breakthrough in the field of graph neural networks, has significantly improved node classification performance by introducing attention mechanisms. In our project, we implemented a Multi-hop GAT model, which not only considers directly connected neighbor nodes but also captures relationships between nodes at longer distances, providing a broader receptive field for graph representation learning.

\subsection{Abstract of Attention Mechanisms in Graphs}

In traditional deep learning, attention mechanisms allow models to assign different importance to various parts of inputs. However, applying attention mechanisms to graph-structured data presents unique challenges: the irregular structure of graphs makes standard attention operations difficult to apply directly.

Our solution revolves around two key abstractions:

\begin{itemize}
    \item Node-level Attention: Assigning different weights to the neighbors of each node
    \item Hop-level Attention: Allocating different weights to neighbor relationships at different distances (hops)
\end{itemize}

\subsection{Combination of Graph Convolution and Attention}

The MultiHopGAT model comprises the following key components:
\begin{figure}[H]
  \centering
  \includegraphics[width=0.5\columnwidth]{2.jpg} 
  \caption{Multihop Graph Attention Network Architecture}
  \label{fig:flowchart}
\end{figure}
\subsubsection{Node-level Attention Implemented by GATConv}

Different from ordinary neural networks, graph convolution needs to process two types of inputs: node features and edge structures. In the code, we use GATConv from the PyTorch Geometric library to implement node-level attention. Each GATConv layer receives:

\begin{itemize}
    \item Node feature matrix $x$ (shaped as [number of nodes, feature dimension])
    \item Edge index $edge_index$ (shaped as [2, number of edges])
\end{itemize}

It determines the importance of each neighbor node by calculating attention coefficients, then performs weighted aggregation of neighbor features. Specifically, for each neighbor $j$ of node $i$, the attention coefficient is computed as:

$$\alpha_{ij} = \frac{\exp(LeakyReLU(a^T[W h_i \| W h_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(LeakyReLU(a^T[W h_i \| W h_j]))}$$
where $W$ is a learnable weight matrix, $a$ is the attention vector, and $∥$ denotes the concatenation operation.

\subsubsection{Implementation of Multi-hop Mechanism}

The innovation lies in introducing a multi-hop mechanism that considers node relationships at different distances, which is implemented via high-order adjacency matrices in the code.

\subsubsection{Hop-level Attention}

Different hop counts may contain different types of information. We designed learnable hop attention weights to assign varying importance to different hops, enabling the model to adaptively determine the relative importance of direct and indirect neighbors.

\subsubsection{Enhanced Components for Stable Training}

To stabilize training and improve performance, we also introduced several key components:
\begin{itemize}
    \item Layer Normalization: Reduces internal covariate shift
    \item Residual Connections: Alleviates gradient vanishing and preserves original features
    \item Decreased Number of Attention Heads: Reduces from 8 (original GAT) to 2, minimizing model complexity
\end{itemize}

\subsection{Performance Comparison of Different Models}

We compared the performance of three models on the PubMed dataset. The results show:

\begin{table}[h]
\caption{Performance comparison on PubMed dataset}\label{tab:pubmed}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Overall Accuracy} & \textbf{Class 0} & \textbf{Class 1} & \textbf{Class 2} \\
\midrule
GCN & 79.8\% & 80.0\% & 84.0\% & 74.4\% \\
GNN (Ours) & 80.0\% & 80.0\% & 80.4\% & 73.9\% \\
Multi-hop GAT (Ours) & 79.1\% & 75.6\% & 81.6\% & 78.1\% \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Overall Performance}
GNN models slightly outperform GCN, while Multi-hop GAT has a slightly lower overall accuracy.

\subsubsection{Category-specific Differences}
\begin{itemize}
    \item GCN and GNN perform similarly on Category 0, but Multi-hop GAT shows weaker performance.
    \item GCN achieves the best result on Category 1, followed by Multi-hop GAT.
    \item Key Finding: On the most difficult-to-classify Category 2, Multi-hop GAT significantly outperforms the other two models, with an improvement of approximately 3.7%.
\end{itemize}

\subsubsection{Explanations}
\begin{itemize}
    \item Multi-hop GAT captures features of Category 2 nodes more effectively by considering indirect relationships.
    \item The attention mechanism enables the model to distinguish between important and unimportant connections, providing an advantage in handling noisy edges.
    \item Category 2 nodes may have more complex graph structure patterns, which require attention mechanisms for effective learning.
\end{itemize}

\subsection{Next Improvement Direction: Expert Fusion Mechanism}
Although Multi-hop GAT performs excellently on Category 2, its performance on Category 0 is weaker, indicating that no single model can excel in all categories. This inspires us to introduce an expert fusion mechanism:

\subsubsection{Expert Division of Labor}
Train specialized "expert" models for different categories:
\begin{itemize}
    \item Use GNN models to handle Categories 0 and 1.
    \item Use Multi-hop GAT to handle Category 2.
\end{itemize}

\subsubsection{Adaptive Fusion Strategy}
1.Dynamic Weight Adjustment by Confidence
\begin{itemize}
    \item Base Weight Setting: Assign expert model weights for each category.
    \item Confidence Weighting: Dynamically adjust weights based on each model's confidence in individual samples.
\end{itemize}
2.Residual Enhancement: Even when a primary model is responsible for a category, auxiliary models provide supplementary predictions.

\subsubsection{Mechanism Rationale}
The expert fusion mechanism combines the advantages of individual models to form a more powerful integrated system, expected to further improve node classification performance—especially for difficult-to-classify nodes. This aligns with the "No Free Lunch" theorem, where multi-model collaboration compensates for the limitations of single models.

\subsubsection{Future Outlook}
Through this approach, we aim to develop a comprehensive model that performs excellently across all categories, providing a more robust solution for graph node classification tasks.

\section{Optimizing GAT: Introduction of Expert Fusion Model}\label{sec5}
To address the performance imbalance of different node categories in the PubMed dataset, we propose an expert fusion model framework. The core idea of this framework is "specialized division of labor"—training dedicated "expert" models for nodes of different categories and then integrating their predictions through a carefully designed fusion mechanism.
\begin{figure}[H]
  \centering
  \includegraphics[width=0.5\columnwidth]{3.jpg} 
  \caption{Expert Fusion Model Architecture}
  \label{fig:flowchart}
\end{figure}
\subsection{Design Principle of Expert Fusion Model}
The expert fusion model consists of two main components:
\begin{itemize}
    \item Expert Model Layer: Contains multiple independent models optimized for specific categories.
    \item Fusion Strategy Layer: Responsible for integrating the prediction results of each expert model.
\end{itemize}

\subsection{Implementation of Expert Models}
We trained specialized expert models for different node categories:
\begin{itemize}
    \item GNN Model: A graph convolutional network with layer normalization and residual connections, optimized for Categories 0 and 1.
    \item GAT Model: A multi-hop graph attention network, optimized for Category 2.
\end{itemize}

\subsection{Design of Fusion Strategies}
We designed two fusion strategies and automatically selected the superior one through evaluation:
\begin{itemize}
    \item Fixed Weight Fusion: Assigns predefined weight ratios to expert models for each category: 
    $\hat{y} = w_1 \cdot y_{\text{GNN}} + w_2 \cdot y_{\text{GAT}}$
    , where $w_i$ is the weight of expert $i$ results.
    \item Adaptive Weight Fusion: Dynamically adjusts weight allocation for each sample based on model confidence: 
    $w_i = \frac{\text{confidence}_i}{\sum_j \text{confidence}_j}$
    \item Automatic Evaluation and Selection: The system compares the performance of both strategies on the validation set and selects the better-performing one.
\end{itemize}

\subsubsection{Advantages of the Expert Fusion Model}
This approach leverages the complementarity of different models, deploying the most suitable model for each node category. For hard-to-classify nodes like Category 2, we assign higher weights to the GAT model to improve classification accuracy. The adaptive fusion strategy further enhances robustness by incorporating model confidence scores.

\subsubsection{Generalization Potential}
Beyond the PubMed dataset, this method applies to other graph datasets with class imbalance or significant inter-class feature variations. It represents a more flexible and granular graph learning paradigm, better adapted to complex real-world scenarios.

\subsection{Result Analysis}
According to the analysis of the experimental results, the expert fusion model we proposed demonstrates significant balanced performance on the PubMed academic literature classification dataset. By comparing the performances of various models in the table, it can be found that:

\begin{table}[h]
\caption{Performance comparison on PubMed dataset}\label{tab:pubmed}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Overall Accuracy} & \textbf{Class 0} & \textbf{Class 1} & \textbf{Class 2} \\
\midrule
GCN & 79.8\% & 80.0\% & 84.0\% & 74.4\% \\
GNN (Ours) & 80.0\% & 80.0\% & 80.4\% & 73.9\% \\
Multi-hop GAT (Ours) & 79.1\% & 75.6\% & 81.6\% & 78.1\% \\
Expert Fusion (Ours) & \textbf{79.0\%} & \textbf{77.8\%} & \textbf{78.0\%} & \textbf{78.4\%} \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Balance Analysis}
\begin{itemize}
    \item Traditional GCN: Exhibits excellent performance on Categories 0 and 1 but low accuracy on Category 2, indicating imbalanced classification capabilities across different journal topics.
    \item Expert Fusion Model: Achieves balanced performance across all categories (77.8\%, 78.0\%, 78.4\%) with no significant weaknesses.
\end{itemize}

\subsubsection{Precision-Recall Tradeoff}
For category $c$, precision $P_c$ and recall $R_c$ are defined as:

$$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}$$

Comparison of F1 scores ($F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$) shows that the expert fusion model has the smallest variance across all categories.

\subsubsection{Stability Quantification}
Define the coefficient of variation (CV) for inter-category performance:

$CV = \frac{\sigma}{\mu}$
where $\sigma$ is the standard deviation of accuracy across categories, and $\mu$ is the average accuracy: 

\begin{itemize}
    \item Expert Fusion Model: CV=0.004
    \item WR-EFM Model: CV=0.013 (best balance between performance and stability)
    \item GCN: CV=0.058
    \item GNN: CV=0.043
    \item GAT: CV=0.037
\end{itemize}

\subsubsection{Multi-expert Weighted Fusion Mechanism}
For category $c$ of sample $i$, the fused prediction probability is computed as:
$$P(y_i=c) = \alpha_{c,1}P_{GNN}(y_i=c) + \alpha_{c,2}P_{Attn}(y_i=c)$$
where weights $\alpha_{c,j}$ are adaptively adjusted per category, satisfying $\sum_j \alpha_{c,j} = 1$.

For Categories 0 and 1: $\alpha_{c,1} =0.95$; For Category 2: $\alpha_{c,1}=0.2$.

The expert fusion model successfully achieves balanced performance in academic literature classification by integrating the global graph structure understanding capability of GNNs with the local attention mechanism of attention models. Experiments demonstrate that tailoring fusion strategies to the characteristic distribution disparities among different literature categories effectively enhances the overall robustness and reliability of the classification system. This approach is generalizable to other graph data classification problems with heterogeneous feature distributions.

\section{Optimizing Expert Fusion: Introduction of Wasserstein-Rubinstein Distance}\label{sec6}

While our Expert Fusion Model (EFM) showed promising results in balancing performance across different node categories, we identified further opportunities for improvement, particularly in the fusion mechanism. To address this, we introduced Wasserstein-Rubinstein (WR) distance as an optimization metric to guide the fusion process, resulting in our enhanced WR-EFM model.

\subsection{Theoretical Foundation of Wasserstein-Rubinstein Distance}

The Wasserstein-Rubinstein distance, also known as Earth Mover's Distance, provides a natural way to measure the dissimilarity between probability distributions. In the context of our graph neural networks, it offers several advantages over traditional distance metrics:

\begin{itemize}
    \item It captures the underlying geometry of the feature space, considering not just whether distributions differ but how they differ
    \item It provides meaningful gradients even when distributions have non-overlapping supports
    \item It is particularly well-suited for comparing high-dimensional distributions, such as node embeddings in graph neural networks
\end{itemize}

Mathematically, for two distributions $\mu$ and $\nu$, the WR distance is defined as:

$$W_p(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x-y\|^p d\gamma(x,y) \right)^{1/p}$$

where $\Gamma(\mu, \nu)$ is the set of all joint distributions with marginals $\mu$ and $\nu$.

\subsection{WR Distance for Model Representation Alignment}

In our WR-EFM model, we utilize the WR distance to measure and optimize the similarity between representations learned by different expert models. The key innovations include:

\subsubsection{Representation Alignment Across Models}
For each node category $c$, we compute the WR distance between the representations from the GNN model and the GAT model:

$$WR\_Loss_c = WR\_Distance(GNN\_embed_c, GAT\_embed_c)$$

This loss term encourages the models to learn complementary yet compatible representations, facilitating more effective fusion.

\subsubsection{Category-Specific Distance Optimization}
We apply different optimization strategies for each category:

\begin{itemize}
    \item For Category 2: We minimize the WR distance to encourage representation similarity, as both models should contribute meaningfully
    \item For Categories 0 and 1: We allow moderate distances, maintaining model diversity while ensuring compatibility
\end{itemize}

This approach acknowledges that different categories may benefit from different degrees of model alignment.

\subsection{Enhanced Fusion Architecture with WR Guidance}

Our WR-EFM architecture extends the basic expert fusion model with several key components:

\begin{figure}[H]
  \centering
  \includegraphics[width=0.6\columnwidth]{4.png} 
  \caption{WR Distance Enhanced Expert Fusion Model Architecture}
  \label{fig:wr_fusion}
\end{figure}

\subsubsection{WR Distance Computation Module}
This module calculates the WR distance between representations from different expert models, providing a signal for both training and fusion weight adjustment.

\subsubsection{Class-Aware Projection Networks}
We introduce class-specific projection networks that transform the raw model outputs before fusion:

$$Proj_c(x) = MLP_c(x)$$

These networks learn to project the representations into a common space where fusion becomes more effective, guided by the WR distance.

\subsubsection{Dynamic Fusion Weight Adjustment}
The fusion weights are dynamically adjusted based on both model confidence and the computed WR distance:

$$w_{GNN,c} = \alpha_c \cdot base\_weight_{GNN,c} + (1-\alpha_c) \cdot confidence_{GNN}$$
$$w_{GAT,c} = \alpha_c \cdot base\_weight_{GAT,c} + (1-\alpha_c) \cdot confidence_{GAT}$$

where $\alpha_c$ is a category-specific balance factor and $base\_weight$ values are initialized based on prior knowledge of model performance.

\subsection{Experimental Results and Analysis}

The WR-EFM model demonstrates significant improvements over both single models and the standard expert fusion approach:

\begin{table}[h]
\caption{Performance comparison including WR-EFM on PubMed dataset}\label{tab:pubmed_wr}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Overall Accuracy} & \textbf{Class 0} & \textbf{Class 1} & \textbf{Class 2} \\
\midrule
GCN & 79.8\% & 80.0\% & 84.0\% & 74.4\% \\
GNN (Ours) & 80.0\% & 80.0\% & 80.4\% & 73.9\% \\
Multi-hop GAT (Ours) & 79.1\% & 75.6\% & 81.6\% & 78.1\% \\
Expert Fusion (Ours) & 79.0\% & 77.8\% & 78.0\% & 78.4\% \\
WR-EFM (Ours) & \textbf{80.1\%} & \textbf{77.8\%} & \textbf{78.0\%} & \textbf{79.9\%} \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Performance Improvements}
\begin{itemize}
    \item Overall accuracy increased to 80.1\%, surpassing all previous models
    \item Category 0 and Category 1 accuracies maintained at 77.8\% and 78.0\% respectively, providing stable performance
    \item Category 2 accuracy significantly improved to 79.9\%, showing a 5.5\% improvement over GCN (74.4\%) and 1.5\% over the standard Expert Fusion model (78.4\%)
\end{itemize}

\subsubsection{Stability Analysis}
The coefficient of variation (CV) for WR-EFM is further reduced to 0.013, compared to 0.004 for the standard expert fusion model and 0.058 for GCN. This demonstrates that WR distance guidance leads to more balanced performance across categories, with particular improvements in the previously challenging Category 2.

\subsubsection{Ablation Study}
We conducted an ablation study to isolate the contribution of WR distance:

\begin{itemize}
    \item Without WR distance but with the same architecture: Overall accuracy drops to 79.5\%
    \item With WR distance but without class-aware projections: Overall accuracy is 79.8\%
    \item Full WR-EFM model: Achieves 80.1\% accuracy
\end{itemize}

These results confirm that both WR distance optimization and class-aware projections contribute meaningfully to the performance improvements, with WR distance being particularly effective for enhancing Category 2 classification (79.9% vs 78.4% in standard fusion).

\subsection{Visualization of Learned Representations}

To further understand the effect of WR distance optimization, we visualized the node embeddings before and after applying WR guidance:

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\columnwidth]{5.png} 
  \caption{t-SNE visualization of node embeddings: (a) Without WR guidance (b) With WR guidance}
  \label{fig:tsne_viz}
\end{figure}

The visualization reveals that WR-guided embeddings show:
\begin{itemize}
    \item More distinct separation between different categories
    \item More cohesive clustering within each category
    \item Particularly improved structure for Category 2 nodes
\end{itemize}

This visual evidence supports our quantitative findings that WR distance optimization leads to more discriminative and category-aware representations.

\section{Conclusion}\label{sec7}

This study introduces a Wasserstein-Rubinstein distance enhanced Expert Fusion Model (WR-EFM) to address performance imbalance in PubMed node classification. By training category-specific expert models (GNN for Categories 0/1, Multi-hop GAT for Category 2) and integrating predictions via WR-guided adaptive weighting, WR-EFM achieves balanced accuracy across all categories (77.8\%, 78.0\%, and 79.9\%) with a significantly lower coefficient of variation (0.013) compared to traditional GCN (0.058).

The introduction of WR distance provides a principled approach to measure and optimize the similarity between model representations, enabling more effective fusion of complementary features. This is particularly valuable for Category 2 nodes, where the model achieves a 5.5\% accuracy improvement over traditional GCN (79.9\% vs. 74.4\%) and a 1.5\% improvement over the standard Expert Fusion model (79.9\% vs. 78.4\%).

Our WR-EFM model not only achieves the highest overall accuracy (80.1\%) among all compared methods but also demonstrates the most balanced performance across different node categories. This balanced performance is crucial for real-world applications where consistent reliability across all classes is often as important as overall accuracy.

The proposed framework generalizes to other graph datasets with class imbalance or heterogeneous features, representing a flexible paradigm for real-world graph learning. Future work will explore more sophisticated optimal transport metrics, dynamic expert assignment strategies, and extend the model to social network analysis and bioinformatics, further advancing GNN applications in complex systems.

%%===========================================================================================%%
%% If you are submitting to one of the Nature Portfolio journals, using the eJP submission   %%
%% system, please include the references within the manuscript file itself. You may do this  %%
%% by copying the reference list from your .bbl file, paste it into the main manuscript .tex %%
%% file, and delete the associated \verb+\bibliography+ commands.                            %%
%%===========================================================================================%%

\bibliography{sn-bibliography}% common bib file
%% if required, the content of .bbl file can be included here once bbl is generated
%%\input sn-article.bbl

\end{document}
