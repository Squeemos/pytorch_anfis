import torch
from torch import nn
from torch.nn import functional as F


@torch.jit.script
def gaussian(x, centers, widths):
    return torch.exp((-(x - centers)**2) / (2 * widths**2))


@torch.jit.script
def triangular(x, lefts, centers, rights, zero):
    return torch.fmax(
        torch.fmin(
            (x - lefts) / (centers - lefts), (rights - x) / (rights - centers)
        ),
        zero,
    )


@torch.jit.script
def difference_of_sigmoid(x, lefts, rights):
    return F.sigmoid(x - lefts) - F.sigmoid(x - rights)


class AnfisLayer(nn.Module):
    """
        Module for an AnfisLayer in PyTorch
    """
    def __init__(
            self,
            in_dim,
            n_rules=8,
            membership_type="Gaussian",
            normal_dis_factor=2.0,
            order=1,
            normalize_rules=True,
    ):
        """
in_dim:             The number of inputs for the AnfisLayer*
n_rules:            The number of rules for each input
membership_type:    The type of rule to use for the AnfisLayer
normal_dis_factor:  What to multiply the original position of the rules
order:              The order of the AnfisLayer.
                    0 means no additional parameters or biases are used.
                    1 means there are additional parameters and biases
normalize_rules:    Whether to normalize the rules or not.
                    If encountering NaNs or infs, try not normalizing the
                    rules as it may lead to numerical instability

*NOTE: The number of input dimensions and output dimensions match.
       This is because the AnfisLayer will be used at the end of the network.
       If you want to have the number of inputs be different than the number
       of outputs, use another layer before the AnfisLayer to increase the
       complexity of the network.
"""
        super(AnfisLayer, self).__init__()

        # Membership functions
        if membership_type == "Gaussian":
            # Means (centers) and Standard Deviation (widths)
            self.register_parameter(
                "centers",
                nn.Parameter(torch.randn(in_dim, n_rules) * normal_dis_factor)
            )
            self.register_parameter(
                "widths",
                nn.Parameter(torch.rand(in_dim, n_rules) * normal_dis_factor)
            )

            self.membership_function = self.gaussian_membership_function
            self.multifunction = False
        elif membership_type == "Triangular":
            # Centers (centers), and left/right points (center +/- width)
            self.register_parameter(
                "centers",
                nn.Parameter(torch.randn(in_dim, n_rules) * normal_dis_factor)
            )
            self.register_parameter(
                "left_widths",
                nn.Parameter(torch.rand(in_dim, n_rules) * normal_dis_factor)
            )
            self.register_parameter(
                "right_widths",
                nn.Parameter(torch.rand(in_dim, n_rules) * normal_dis_factor)
            )
            self.register_buffer(
                "common_zero",
                torch.tensor(0.0)
            )

            self.membership_function = self.triangular_membership_function
            self.multifunction = False
        elif membership_type == "Difference of Sigmoid":
            self.register_parameter(
                "lefts",
                nn.Parameter(torch.randn(in_dim, n_rules) * normal_dis_factor)
            )
            self.register_parameter(
                "rights",
                nn.Parameter(self.lefts.clone() + torch.rand(in_dim, n_rules))
            )

            self.membership_function = self.difference_of_sigmoid_membership_function  # noqa: E501
            self.multifunction = False
        elif membership_type == "Multifunction":
            self.multifunction = True
        else:
            # TODO: Add in other membership functions
            raise NotImplementedError(
                f"AnfisLayer with membership type <{self.membership_type}> is not supported"  # noqa: E501
            )

        # Higher order ANFIS functions
        if order == 0:
            pass
        elif order == 1:
            # Learning parameters
            # Jang paper
            self.register_parameter(
                "params",
                nn.Parameter(torch.randn(in_dim, n_rules) * normal_dis_factor)
            )
            self.register_parameter(
                "biases",
                nn.Parameter(torch.randn(in_dim, n_rules) * normal_dis_factor)
            )

        # Setup the members for later referencing
        self.membership_type = membership_type
        self.n_rules = n_rules
        self.normalize_rules = normalize_rules
        self.in_dim = in_dim
        self.order = order

    def gaussian_membership_function(self, x):
        return gaussian(x, self.centers, self.widths)

    def triangular_membership_function(self, x):
        batch_size = x.shape[0]
        # Get right/left points of the triangles
        # (batch_size, in_dim, n_rules)
        lefts = (self.centers - self.left_widths).expand(batch_size, -1, -1)
        rights = (self.centers + self.right_widths).expand(batch_size, -1, -1)
        centers = self.centers.expand(batch_size, -1, -1)

        # Perform the membership function
        # (batch_size, in_dim, n_rules)
        return triangular(x, lefts, centers, rights, self.common_zero)

    def difference_of_sigmoid_membership_function(self, x):
        return difference_of_sigmoid(x, self.lefts, self.rights)

    def forward(self, x):
        # Expand for broadcasting with rules
        # (batch_size, in_dim, n_rules)
        x = x.unsqueeze(-1).expand(-1, -1, self.n_rules)

        if self.multifunction:
            pass
        else:
            return self.membership_to_output(x)

    def membership_to_output(self, x):
        # Fuzzification
        # (batch_size, in_dim, n_rules)
        membership = self.membership_function(x)

        # Normalization
        # (batch_size, in_dim, n_rules)
        # If the rules are going to be normalized
        if self.normalize_rules:
            rule_evaluation = membership / membership.sum(dim=-1, keepdim=True)  # noqa: E501
        else:
            rule_evaluation = membership

        # Multiply the input by the learnable parameters and add the biases
        # (batch_size, in_dim, n_rules)
        if self.order == 0:
            defuzz = rule_evaluation
        elif self.order == 1:
            defuzz = x * self.params + self.biases

        # Multiply the rules by the fuzzified input
        # (batch_size, in_dim, n_rules)
        output = rule_evaluation * defuzz

        # Sum the rules
        # (batch_size, in_dim)
        return output.sum(dim=-1)

    def __repr__(self):
        repr_str = super(AnfisLayer, self).__repr__()[:-2] + f"(in_features={self.in_dim}, out_features={self.in_dim})\n"  # noqa: E501
        repr_str += f"  (n_rules): {self.n_rules}\n"

        if self.membership_type == "Gaussian":
            repr_str += f"  (centers): {self.centers.shape, self.centers.dtype}\n"  # noqa: E501
            repr_str += f"  (widths): {self.widths.shape, self.widths.dtype}\n"  # noqa: E501
        elif self.membership_type == "Triangular":
            repr_str += f"  (centers): {self.centers.shape, self.centers.dtype}\n"  # noqa: E501
            repr_str += f"  (left_widths): {self.left_widths.shape, self.left_widths.dtype}\n"  # noqa: E501
            repr_str += f"  (right_widths): {self.right_widths.shape, self.right_widths.dtype}\n"  # noqa: E501

        if self.order == 1:
            repr_str += f"  (learnable_params): {self.params.shape, self.params.dtype}\n"  # noqa: E501
            repr_str += f"  (learnable_biases): {self.biases.shape, self.biases.dtype}\n)"  # noqa: E501

        return repr_str
