import numpy as np
from utility import sigmoid


class BprOptimizer():
    """
    use Bayesian Personalized Ranking for objective loss
    latent_dimen: latent dimension
    alpha: learning rate
    matrix_reg: regularization parameter of matrix
    """
    def __init__(self, latent_dimen, alpha, matrix_reg):
        self.latent_dimen = latent_dimen
        self.alpha = alpha
        self.matrix_reg = matrix_reg

    def init_model(self, dataset):
        """
        initialize matrix using uniform [-0.2, 0.2]
        """
        self.paper_latent_matrix = {}
        self.author_latent_matrix = {}
        for paper_idx in dataset.paper_list:
            self.paper_latent_matrix[paper_idx] = np.random.uniform(-0.2, 0.2,
                                                                    self.latent_dimen)
        for author_idx in dataset.coauthor_list:
            self.author_latent_matrix[author_idx] = np.random.uniform(-0.2, 0.2,
                                                                      self.latent_dimen)

    def update_pp_gradient(self, fst, snd, third):
        """
        SGD inference
        """
        x = self.predict_score(fst, snd, "pp") - \
            self.predict_score(fst, third, "pp")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.author_latent_matrix[snd] - \
                                  self.author_latent_matrix[third]) + \
                    2 * self.matrix_reg * self.author_latent_matrix[fst]
        self.author_latent_matrix[fst] = self.author_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.author_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.author_latent_matrix[snd]
        self.author_latent_matrix[snd]= self.author_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.author_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.author_latent_matrix[third]
        self.author_latent_matrix[third] = self.author_latent_matrix[third] - \
                                           self.alpha * grad_third

    def update_pd_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "pd") - \
            self.predict_score(fst, third, "pd")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.author_latent_matrix[snd] - \
                                  self.author_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.author_latent_matrix[snd]
        self.author_latent_matrix[snd]= self.author_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.author_latent_matrix[third]
        self.author_latent_matrix[third] = self.author_latent_matrix[third] - \
                                           self.alpha * grad_third

    def update_dd_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dd") - \
            self.predict_score(fst, third, "dd")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.paper_latent_matrix[snd] - \
                                  self.paper_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.paper_latent_matrix[snd]
        self.paper_latent_matrix[snd]= self.paper_latent_matrix[snd] - \
                                       self.alpha * grad_snd

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.paper_latent_matrix[third]
        self.paper_latent_matrix[third] = self.paper_latent_matrix[third] - \
                                          self.alpha * grad_third

    def compute_pp_loss(self, fst, snd, third):
        """
        loss includes ranking loss and model complexity
        """
        x = self.predict_score(fst, snd, "pp") - \
             self.predict_score(fst, third, "pp")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[fst],
                                               self.author_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[snd],
                                               self.author_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[third],
                                               self.author_latent_matrix[third])
        return ranking_loss + complexity

    def compute_pd_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "pd") - \
            self.predict_score(fst, third, "pd")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[snd],
                                               self.author_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[third],
                                               self.author_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dd_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dd") - \
            self.predict_score(fst, third, "dd")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[snd],
                                               self.paper_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[third],
                                               self.paper_latent_matrix[third])
        return ranking_loss + complexity

    def predict_score(self, fst, snd, graph_type):
        """
        pp: person-person network
        pd: person-document bipartite network
        dd: doc-doc network
        detailed notation is inside paper
        """
        if graph_type == "pp":
            return np.dot(self.author_latent_matrix[fst],
                          self.author_latent_matrix[snd])
        elif graph_type == "pd":
            return np.dot(self.paper_latent_matrix[fst],
                          self.author_latent_matrix[snd])
        elif graph_type == "dd":
            return np.dot(self.paper_latent_matrix[fst],
                          self.paper_latent_matrix[snd])
