import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def embedding_similarities():
    Entities_Embedding_file = './data/input_embedding_dict/ic/dict_ic.list'
    Entities_Embedding_List = {}
    IdEntiy__Embedding_List = {}
    proteinidlist = []
    proteinnamelist = []
    drugnamelist = []
    drugidlist = []
    with open(Entities_Embedding_file) as f:
        for line in f:
            it = line.strip().split("\t")
            id = int(it[0])
            entity = str(it[1])
            if id > 168:
                proteinidlist.append(id)
                proteinnamelist.append(entity)
                if entity not in Entities_Embedding_List:
                    Entities_Embedding_List[entity] = set()
                Entities_Embedding_List[entity].add(int(id))
                if id not in IdEntiy__Embedding_List:
                    IdEntiy__Embedding_List[id] = set()
                IdEntiy__Embedding_List[id].add(str(entity))
            else:
                drugidlist.append(id)
                drugnamelist.append(entity)
    print(len(proteinnamelist))

    # Load entity embeddings
    entity_embeddings = np.loadtxt(
        './data/input_embedding_dict/ic/entity_embedding_dt_ic.csv',)


    # Your other code here...

    # Assuming embedcosinusDrugslist, embedcosinusProteinslist, and embedcosinusTargetlist
    # contain the indices of drugs, proteins, and targets respectively
    subembedding = entity_embeddings[proteinidlist]
    print(len(subembedding))
    print(subembedding.shape)
    # Calculate cosine similarities between proteins
    similarities_proteins = cosine_similarity(subembedding)

    # Now similarities_proteins contains cosine similarities between proteins
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/cosine_similarities_ic_proteins.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins):
            for j, similarity in enumerate(row):
                protein1 = proteinnamelist[i]
                protein2 = proteinnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from sklearn.metrics.pairwise import euclidean_distances
    similarities_protein_ECLD = euclidean_distances(entity_embeddings[proteinidlist])
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/eucludian_similarities_ic_proteins.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_protein_ECLD):
            for j, similarity in enumerate(row):
                protein1 = proteinnamelist[i]
                protein2 = proteinnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from sklearn.metrics.pairwise import manhattan_distances
    similarities_proteins_mnht = manhattan_distances(entity_embeddings[proteinidlist])
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/manhatan_similarities_ic_proteins.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins_mnht):
            for j, similarity in enumerate(row):
                protein1 = proteinnamelist[i]
                protein2 = proteinnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))
    from scipy.stats import pearsonr
    similarities_proteins_pearson = np.array(
        [[pearsonr(a, b)[0] for b in entity_embeddings[proteinidlist]] for a in entity_embeddings[proteinidlist]])
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/pearson_similarities_ic_proteins.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins_pearson):
            for j, similarity in enumerate(row):
                protein1 = proteinnamelist[i]
                protein2 = proteinnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from sklearn.metrics import jaccard_score
    # Threshold for converting numerical values to binary (0 or 1)
    threshold = 0.5  # You can adjust this threshold based on your data

    # Convert numerical values to binary based on the threshold
    binary_protein_sets = [[int(value > threshold) for value in entity_embeddings[proteinid]] for proteinid in
                           proteinidlist]

    # Calculate Jaccard similarities between proteins
    similarities_proteins_jaccard = np.array(
        [[jaccard_score(set_a, set_b) for set_b in binary_protein_sets] for set_a in binary_protein_sets])
    with open('./data/input_embedding_dict/ic/jaccard_similarities_ic_proteins.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins_jaccard):
            for j, similarity in enumerate(row):
                protein1 = proteinnamelist[i]
                protein2 = proteinnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    # Assuming embedcosinusDrugslist, embedcosinusProteinslist, and embedcosinusTargetlist
    # contain the indices of drugs, proteins, and targets respectively
    subembedding = entity_embeddings[drugidlist]
    print(len(subembedding))
    print(subembedding)
    # Calculate cosine similarities between proteins
    similarities_drugs = cosine_similarity(subembedding)

    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/cosine_similarities_ic_drugs.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_drugs):
            for j, similarity in enumerate(row):
                protein1 = drugnamelist[i]
                protein2 = drugnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from sklearn.metrics.pairwise import euclidean_distances
    similarities_protein_ECLD = euclidean_distances(entity_embeddings[drugidlist])
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/eucludian_similarities_ic_drugs.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_protein_ECLD):
            for j, similarity in enumerate(row):
                protein1 = drugnamelist[i]
                protein2 = drugnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from sklearn.metrics.pairwise import manhattan_distances
    similarities_proteins_mnht = manhattan_distances(entity_embeddings[drugidlist])
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/manhatan_similarities_ic_drugs.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins_mnht):
            for j, similarity in enumerate(row):
                protein1 = drugnamelist[i]
                protein2 = drugnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from scipy.stats import pearsonr
    similarities_proteins_pearson = np.array(
        [[pearsonr(a, b)[0] for b in entity_embeddings[drugidlist]] for a in entity_embeddings[drugidlist]])
    # Save cosine similarities to a text file
    with open('./data/input_embedding_dict/ic/pearson_similarities_ic_drugs.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins_pearson):
            for j, similarity in enumerate(row):
                protein1 = drugnamelist[i]
                protein2 = drugnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))

    from sklearn.metrics import jaccard_score
    # Threshold for converting numerical values to binary (0 or 1)
    threshold = 0.5  # You can adjust this threshold based on your data

    # Convert numerical values to binary based on the threshold
    binary_protein_sets = [[int(value > threshold) for value in entity_embeddings[proteinid]] for proteinid in
                           drugidlist]

    # Calculate Jaccard similarities between proteins
    similarities_proteins_jaccard = np.array(
        [[jaccard_score(set_a, set_b) for set_b in binary_protein_sets] for set_a in binary_protein_sets])
    with open('./data/input_embedding_dict/ic/jaccard_similarities_ic_drugs.txt', 'w') as f:
        # Save drug similarities
        for i, row in enumerate(similarities_proteins_jaccard):
            for j, similarity in enumerate(row):
                protein1 = drugnamelist[i]
                protein2 = drugnamelist[j]
                f.write('{}\t{}\t{}\n'.format(protein1, protein2, similarity))