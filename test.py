import numpy as np
import torch
import unittest
import copy
from LORI import make_low_rank_nn_linear, form_off_diag, LRPBD_from_matrix, LRPBD_iterated_soln, reduce_rank_causal_attn, LORICausalSelfAttention, make_LORI_FC_from_matrix, make_LORI_FC_from_module, LORIMLP, reduce_rank_MLP
from model import CausalSelfAttention, GPTConfig, MLP, Block
  

def make_linear(arr, in_features, out_features, bias):
    original = torch.nn.Linear(in_features=in_features, out_features=out_features, bias = bias)
    new_wts = torch.Tensor( np.array(arr))
    sd = original.state_dict()
    sd['weight'] = new_wts
    original.load_state_dict(sd)
    # print('wts after load state dict')
    # print(original.state_dict()['weight'])
    return original

def make_condition_number_large(original):
    assert isinstance(original, torch.nn.Linear)
    sd = original.state_dict()
    wts = sd['weight']

    must_flip = False
    if wts.shape[1] > wts.shape[0]:
        must_flip = True
        wts = torch.transpose(wts, 0,1)

    U, S, V = torch.linalg.svd(wts, full_matrices = False)
    S[-1] = S[0] * 1e-4
    new_wts = U @ torch.diag(S) @ V
    if must_flip:
        new_wts = torch.transpose(new_wts, 0,1)
    sd['weight'] = new_wts
    original.load_state_dict(sd)
    return original


class TestRankReduce(unittest.TestCase):
    def test_square_nobias(self):
        INPUT_SIZE = 4
        OUTPUT_SIZE = 4
        NEW_RANK = 3

        original = make_linear([[1.0,  0.5, 0.0, 0.0   ],
                                [0.0, -2.0, 0.1, 0.0   ],
                                [0.0,  0.0, 0.1, 0.0   ],
                                [0.0,  0.0, 0.0, 0.0001]], in_features=INPUT_SIZE, out_features = OUTPUT_SIZE, bias=False )

        input = torch.Tensor( np.array([1.0, 1.0, 1.0, 1.0]))
        full_rank_output = original(input)
        low_rank_module = make_low_rank_nn_linear(original, new_rank = NEW_RANK)
        low_rank_output = low_rank_module(input)

        low_rank_weights2 = low_rank_module.state_dict()['weight']
        # print('low rank weights:')
        # print(low_rank_weights2)
        full_rank_wts = original.state_dict()['weight']
        # We can expect wts to be close because condition number was so high/ ignored 
        # singular values that wer so tiny.
        np.testing.assert_allclose(full_rank_wts, low_rank_weights2, atol=1e-3)        
        np.testing.assert_allclose(full_rank_output, low_rank_output, atol=1e-3)

    def test_square_no_bias_keep_full_rank(self):
        INPUT_SIZE = 4
        OUTPUT_SIZE = 4
        NEW_RANK = 4

        original = make_linear([[1.0,  0.5, 0.0, 0.0   ],
                                [0.0, -2.0, 0.1, 0.0   ],
                                [0.0,  0.0, 0.1, 0.0   ],
                                [0.0,  0.0, 0.0, 0.0001]], in_features=INPUT_SIZE, out_features = OUTPUT_SIZE, bias=False )

        input = torch.Tensor( np.array([1.0, 1.0, 1.0, 1.0]))
        full_rank_output = original(input)
        low_rank_module = make_low_rank_nn_linear(original, new_rank = NEW_RANK)
        low_rank_output = low_rank_module(input)

        low_rank_weights2 = low_rank_module.state_dict()['weight']
        # print('low rank weights:')
        # print(low_rank_weights2)
        full_rank_wts = original.state_dict()['weight']
        np.testing.assert_allclose(full_rank_wts, low_rank_weights2, atol=1e-3)        
        np.testing.assert_allclose(full_rank_output, low_rank_output, atol=1e-3)

    # Linear has more inputs than outputs.
    def test_in_gt_out_no_bias_full_rank(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 4
        NEW_RANK = 4
        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        low_rank_module = make_low_rank_nn_linear(original, new_rank = NEW_RANK)
        
        low_rank_output = low_rank_module(input)
        full_rank_output = original(input)
        np.testing.assert_allclose(full_rank_output, low_rank_output, atol=1e-3)

    # Linear has more outputs than inputs.
    def test_out_gt_in_no_bias_full_rank(self):
        INPUT_SIZE = 4
        OUTPUT_SIZE = 8
        NEW_RANK = 4

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0]))

        low_rank_module = make_low_rank_nn_linear(original, new_rank = NEW_RANK)
        low_rank_output = low_rank_module(input)
        full_rank_output = original(input)
        np.testing.assert_allclose(full_rank_output, low_rank_output, atol=1e-3)

    # Check the checker
    def test_detects_rank_loss_no_bias(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 4
        NEW_RANK = 3

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        low_rank_module = make_low_rank_nn_linear(original, new_rank = NEW_RANK)
        low_rank_output = low_rank_module(input)
        full_rank_output = original(input)
        is_close = np.isclose(full_rank_output, low_rank_output, atol=1e-4)
        # There should be difference between low rank output and full rank output
        np.testing.assert_(not is_close.all(), 'Low rank approximation should be detectably different')

    def test_in_gt_out_good_approximation_no_bias(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 4
        NEW_RANK = 3

        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        # Make a version of original with a high condition number so rank reduction
        # will still be a good enough approx to pass np.all_close assertion
        hcn_original = make_condition_number_large(original)
        low_rank_module = make_low_rank_nn_linear(hcn_original, new_rank = NEW_RANK)
        
        low_rank_output = low_rank_module(input)
        full_rank_output = hcn_original(input)
        np.testing.assert_allclose(full_rank_output, low_rank_output, atol=1e-3)

    def test_form_off_diag(self):
        original = torch.ones(size=(8,4))
        expected = torch.Tensor([[0.0,  0.0, 1.0, 1.0 ],
                                 [0.0,  0.0, 1.0, 1.0 ],
                                 [0.0,  0.0, 1.0, 1.0 ],
                                 [0.0,  0.0, 1.0, 1.0 ],
                                 [1.0,  1.0, 0.0, 0.0 ],
                                 [1.0,  1.0, 0.0, 0.0 ],
                                 [1.0,  1.0, 0.0, 0.0 ],
                                 [1.0,  1.0, 0.0, 0.0 ] ])
        off_diag = form_off_diag(original, block_rows=4, block_cols=2)
        np.testing.assert_allclose(expected, off_diag, atol=1e-3)

    def test_LRPBD_square_full(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 8
        NEW_RANK = 8
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 0.0001)

    def test_LRPBD_square_low_rank(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 8
        NEW_RANK = 4
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 0.70)
        self.assertGreater(residual, 0.01)

    def test_LRPBD_in_gt_out_full_rank(self):
        INPUT_SIZE = 12
        OUTPUT_SIZE = 6
        NEW_RANK = 6
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 0.0001)

    def test_LRPBD_in_lt_out_full_rank(self):
        INPUT_SIZE = 6
        OUTPUT_SIZE = 12
        NEW_RANK = 6
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 0.0001)

    def test_LRPBD_in_gt_out_low_rank(self):
        INPUT_SIZE = 12
        OUTPUT_SIZE = 6
        NEW_RANK = 3
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 0.8)
        self.assertGreater(residual, 0.01)

    def test_LRPBD_in_lt_out_low_rank(self):
        INPUT_SIZE = 6
        OUTPUT_SIZE = 12
        NEW_RANK = 3
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 1.0)
        self.assertGreater(residual, 0.001)

    def test_module_LRPBD_square_full(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 8
        NEW_RANK = 8
        BLOCK_SIZE = 2
        
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        lori_fc = make_LORI_FC_from_module(original, new_rank = NEW_RANK, block_size=BLOCK_SIZE, bias=False)

        expected_output = original(input)
        approx_output = lori_fc(input)
        np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)

        # print('original wts:')
        # print(original_wts)
        # approx_wts = lori_fc.left.weight @ lori_fc.right.weight
        # print('approx wts:')
        # print(approx_wts)
        # print('left:')
        # print(lori_fc.left.weight)
        # print('right:')
        # print(lori_fc.right.weight)

    def test_module_LRPBD_in_gt_out_full(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 4
        NEW_RANK = 4
        BLOCK_SIZE = 2
        
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        lori_fc = make_LORI_FC_from_module(original, new_rank = NEW_RANK, block_size=BLOCK_SIZE, bias=False)

        expected_output = original(input)
        approx_output = lori_fc(input)
        np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)

        # Compute what the weights of the approximated nn.Linear module would be
        # to match the approximated weights of the LORI_FC module.
        approx_wts = torch.transpose(torch.transpose(lori_fc.left.weight,0,1) @ torch.transpose(lori_fc.right.weight,0,1) +
                                     torch.block_diag(*lori_fc.diag_params), 0,1)
        np.testing.assert_allclose(original_wts, approx_wts, atol=1e-3)

    def test_module_LRPBD_in_gt_out_full_2(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 4
        NEW_RANK = 8
        BLOCK_SIZE = 2
        
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        lori_fc = make_LORI_FC_from_module(original, new_rank = NEW_RANK, block_size=BLOCK_SIZE, bias=False)

        expected_output = original(input)
        approx_output = lori_fc(input)
        np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)

        # Compute what the weights of the approximated nn.Linear module would be
        # to match the approximated weights of the LORI_FC module.
        approx_wts = torch.transpose(torch.transpose(lori_fc.left.weight,0,1) @ torch.transpose(lori_fc.right.weight,0,1) +
                                     torch.block_diag(*lori_fc.diag_params), 0,1)
        np.testing.assert_allclose(original_wts, approx_wts, atol=1e-3)


    def test_module_LRPBD_out_gt_in_full(self):
        INPUT_SIZE = 4
        OUTPUT_SIZE = 8
        NEW_RANK = 4
        BLOCK_SIZE = 2
        
        input = torch.Tensor(np.array([1.0, 1.0, 1.0, 1.0]))

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        lori_fc = make_LORI_FC_from_module(original, new_rank = NEW_RANK, block_size=BLOCK_SIZE, bias=False)

        expected_output = original(input)
        approx_output = lori_fc(input)
        np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)

        # Compute what the weights of the approximated nn.Linear module would be
        # to match the approximated weights of the LORI_FC module.
        approx_wts = torch.transpose(torch.transpose(lori_fc.left.weight,0,1) @ torch.transpose(lori_fc.right.weight,0,1) +
                                     torch.block_diag(*lori_fc.diag_params), 0,1)
        np.testing.assert_allclose(original_wts, approx_wts, atol=1e-3)

    def test_iterated_soln(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 8
        NEW_RANK = 4
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)

        # print('approximated:')
        # print(approximated)
        # print('original:')
        # print(original_wts)
        # print('residual:')
        # print(residual)
        self.assertLess(residual, 0.80)
        self.assertGreater(residual, 0.001)

    def test_iterated_soln_in_gt_out_full(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 4
        NEW_RANK = 4
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)

        # print('approximated:')
        # print(approximated)
        # print('original:')
        # print(original_wts)
        # print('residual:')
        # print(residual)
        self.assertLess(residual, 0.001)

    def test_iterated_soln_out_gt_in_full(self):
        INPUT_SIZE = 4
        OUTPUT_SIZE = 8
        NEW_RANK = 4
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)

        # print('approximated:')
        # print(approximated)
        # print('original:')
        # print(original_wts)
        # print('residual:')
        # print(residual)
        self.assertLess(residual, 0.001)

    def test_iterated_soln_in_gt_out(self):
        INPUT_SIZE = 16
        OUTPUT_SIZE = 8
        NEW_RANK = 4
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)
        self.assertLess(residual, 0.80)
        self.assertGreater(residual, 0.001)

    def test_iterated_soln_out_gt_in(self):
        INPUT_SIZE = 8
        OUTPUT_SIZE = 16
        NEW_RANK = 4
        BLOCK_SIZE = 2

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = torch.linalg.matrix_norm(original_wts - approximated)

        self.assertLess(residual, 0.80)
        self.assertGreater(residual, 0.00)

    def test_iterated_square(self):
        INPUT_SIZE = 100
        OUTPUT_SIZE = 100
        NEW_RANK = 10
        BLOCK_SIZE = 5

        original = torch.nn.Linear(in_features=INPUT_SIZE, out_features=OUTPUT_SIZE, bias = False)
        original_wts = original.state_dict()['weight']

        left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(original_wts, 
                                                                         new_rank = NEW_RANK, 
                                                                         block_size=BLOCK_SIZE)

        approximated = left_matrix @ right_matrix + torch.block_diag(*block_diag_params)
        residual = original_wts - approximated
        # for i in range(residual.shape[0]):
        #     for j in range(residual.shape[1]):
        #         print('{0},{1}: {2}'.format(i,j,residual[i,j]))
        error = torch.linalg.matrix_norm(residual) / torch.linalg.matrix_norm(original_wts)
        self.assertLess(error, 0.9)
        self.assertGreater(error, 0.00)

    def test_causal_attn_iterated_soln_full(self):
        full_config = GPTConfig()
        full_config.bias=False
        B = 2
        T = full_config.block_size
        C = full_config.n_embd
        input = torch.rand(size=(B, T, C))
        ca = CausalSelfAttention(full_config)
        expected_output = ca(input)

        lori_config = copy.copy(full_config)
        lori_config.lori = True
        lori_config.n_fc_bottleneck = full_config.n_embd
        lori_config.n_fc_diagblock = 8
        lori_config.n_k=full_config.n_embd
        lori_config.n_q=full_config.n_embd
        lori_config.n_v=full_config.n_embd

        lori_ca = LORICausalSelfAttention(lori_config)
        reduce_rank_causal_attn(ca, lori_ca, lori_config)

        approx_output = lori_ca(input)        
        # print('expected output - approx output:')
        # print(expected_output-approx_output)
        # error = torch.linalg.matrix_norm(expected_output-approx_output)
        # print('error:{0}'.format(error))
        np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)        


    def test_causal_attn_iterated_soln_low_rank(self):
        full_config = GPTConfig()
        full_config.bias=False
        B = 2
        T = full_config.block_size
        C = full_config.n_embd
        input = torch.rand(size=(B, T, C))
        ca = CausalSelfAttention(full_config)
        expected_output = ca(input)

        lori_config = copy.copy(full_config)
        lori_config.lori = True
        lori_config.n_fc_bottleneck = full_config.n_embd // 16
        lori_config.n_fc_diagblock = 8
        lori_config.n_k=full_config.n_embd // 16
        lori_config.n_q=full_config.n_embd // 16
        lori_config.n_v=full_config.n_embd // 16

        lori_ca = LORICausalSelfAttention(lori_config)
        reduce_rank_causal_attn(ca, lori_ca, lori_config)

        approx_output = lori_ca(input)        
        # print('expected output:')
        # print(expected_output)
        # print('approx output:')
        # print(approx_output)
        # error = torch.linalg.matrix_norm(expected_output-approx_output)
        # print('error:{0}'.format(error))
        # np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)
        error = torch.mean(torch.abs(expected_output-approx_output))
        self.assertGreater(error, 0.001)
        self.assertLess(error, 1.0)
        # magnitude = torch.mean(torch.abs(expected_output))

    def test_MLP_iterated_soln_full(self):
        full_config = GPTConfig()
        full_config.bias=False
        B = 2
        T = full_config.block_size
        C = full_config.n_embd
        input = torch.rand(size=(B, T, C))
        mlp = MLP(full_config)
        expected_output = mlp(input)

        lori_config = copy.copy(full_config)
        lori_config.lori = True
        lori_config.n_fc_bottleneck = full_config.n_embd
        lori_config.n_fc_diagblock = 8
        lori_config.n_k=full_config.n_embd
        lori_config.n_q=full_config.n_embd
        lori_config.n_v=full_config.n_embd

        lori_mlp = LORIMLP(lori_config)
        reduce_rank_MLP(mlp, lori_mlp, lori_config)

        approx_output = lori_mlp(input)
        # print('expected output - approx output:')
        # print(expected_output-approx_output)
        # error = torch.linalg.matrix_norm(expected_output-approx_output)
        # print('error:{0}'.format(error))
        np.testing.assert_allclose(expected_output, approx_output, atol=1e-3)        


    def test_MLP_iterated_soln_low_rank(self):
        full_config = GPTConfig()
        full_config.bias=False
        B = 2
        T = full_config.block_size
        C = full_config.n_embd
        input = torch.rand(size=(B, T, C))
        mlp = MLP(full_config)
        expected_output = mlp(input)

        lori_config = copy.copy(full_config)
        lori_config.lori = True
        lori_config.n_fc_bottleneck = full_config.n_embd // 4
        lori_config.n_fc_diagblock = 4
        lori_config.n_k=full_config.n_embd // 4
        lori_config.n_q=full_config.n_embd // 4
        lori_config.n_v=full_config.n_embd // 4

        lori_mlp = LORIMLP(lori_config)
        reduce_rank_MLP(mlp, lori_mlp, lori_config)

        # approx_output = lori_mlp(input)
        # print('expected output - approx output:')
        # print(expected_output-approx_output)
        # error = torch.linalg.matrix_norm(expected_output-approx_output)
        # output_mag = torch.linalg.matrix_norm(expected_output)
        # approx_output_mag = torch.linalg.matrix_norm(approx_output)
        # print('error:{0}, output_mag:{1} approx mag:{2}'.format(error, output_mag, approx_output_mag))
        cos_sim = torch.nn.CosineSimilarity(dim=2)
        sim = cos_sim(expected_output, approx_output)
        
        print('cos similarity"{0}'.format(sim))
        is_greater = np.greater(sim, 0.0)
        np.testing.assert_(is_greater.all())



if __name__ == '__main__':
    with torch.no_grad():
        unittest.main()
