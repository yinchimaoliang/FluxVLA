# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GR00T N1.7 RoboDojo relative-joint training and FluxThemis evaluation.

Matches the ARX-X5 reference action contract: both seven-dimensional arms
(including grippers) are relative to the current raw state. Sixteen valid
steps are normalized using per-offset relative min/max bounds, then padded
to the model's native 40 x 132 target shape. Inference returns 16 steps.
Start from the original GR00T checkpoint, not the legacy absolute-action run.
The five-epoch budget is an initial validation cap, not a source score claim.
"""

_base_ = ['./gr00tn17_qwen3vl_2b_libero_10_full_finetune.py']

_ROBODOJO_DATA_ROOT = './datasets/RoboDojo_lerobot_v21_video'
_STATISTIC_NAME = 'robodojo_arx_x5_relative'
_EMBODIMENT_KEY = 'new_embodiment'
# Official N1.7 NEW_EMBODIMENT projector slot.
_EMBODIMENT_ID = 10
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'
_ACTION_WINDOW_SIZE = 16
_RELATIVE_ACTION_MASK = [True] * 14
_N17_LAYOUT = (('left_arm', 7), ('right_arm', 7))

# Recomputed from all 3,500 episodes / 1,859,602 raw frames. Relative
# bounds use all 1,807,102 complete 16-step windows; no repeated tail frames.
# Source: XPolicyLab GR00T_N17 arx_x5_config + N1.7 RelativeActionLoader.
# Relative normalization uses min/max even when use_percentiles=True;
# that setting selects q01/q99 only for the absolute state statistics.
_STATE_STATISTICS = {
    'min': [
        -1.4863648414611816, -0.2586335241794586, -0.06850092858076096,
        -2.2642147541046143, -1.4876768589019775, -3.0168392658233643,
        -3.212450869184004e-17, -1.4924354553222656, -0.27809593081474304,
        -0.18774886429309845, -2.5354628562927246, -3.0663280487060547,
        -3.0486762523651123, -3.212450869184004e-17
    ],
    'max': [
        1.7537026405334473, 3.0546703338623047, 3.8525187969207764,
        2.0974574089050293, 1.8103265762329102, 3.1366536617279053, 1.0,
        1.6732014417648315, 3.3797459602355957, 4.252756595611572,
        2.5287232398986816, 3.1440579891204834, 3.0381109714508057, 1.0
    ],
    'mean': [
        -0.20040483418845204, 0.9260308650671458, 0.6881893789087474,
        -0.3400484544283471, 0.06579394032953159, 0.004003331131509558,
        0.771446548553906, 0.17299834959584776, 0.8224941837716736,
        0.6196513047400859, -0.3604178625374315, -0.06267071413702419,
        0.0037534824507240434, 0.7860054714056368
    ],
    'std': [
        0.34305506136703257, 0.8880398716756615, 0.7302413867018247,
        0.6487680858145891, 0.2989095916390101, 0.584300843989697,
        0.34316664624786114, 0.31989806135013626, 0.8736479432260295,
        0.7135350385777451, 0.6027223159771133, 0.2836475966439172,
        0.5368540820848867, 0.3406560080480477
    ],
    'q01': [
        -1.051365569829941, -3.52302449637288e-14, 1.6996893991849812e-16,
        -1.5984010362625123, -0.6003412520885467, -1.6678147149085998, 0.0,
        -0.4500492787361145, -2.5859282299029243e-14, 1.194697284288627e-16,
        -1.6398817586898804, -1.263367258310318, -1.760098042488098, 0.0
    ],
    'q99': [
        0.5431358617544174, 2.495765209197998, 2.492974226474762,
        1.323737324476242, 1.2496635341644287, 1.7391636216640465, 1.0,
        1.0814965963363647, 2.416655488014221, 2.346989154815674,
        1.1415718960762022, 0.5208872479200363, 1.4889542925357817, 1.0
    ],
    'count':
    1859602
}

_ABSOLUTE_ACTION_STATISTICS = {
    'min': [
        -1.4863648414611816, -0.2586335241794586, -0.06850092858076096,
        -2.2642147541046143, -1.4876768589019775, -3.0168392658233643,
        -3.212450869184004e-17, -1.4924354553222656, -0.27809593081474304,
        -0.18774886429309845, -2.5354628562927246, -3.0663280487060547,
        -3.0486762523651123, -3.212450869184004e-17
    ],
    'max': [
        1.7537026405334473, 3.0546703338623047, 3.8525187969207764,
        2.0974574089050293, 1.8103265762329102, 3.1366536617279053, 1.0,
        1.6732014417648315, 3.3797459602355957, 4.252756595611572,
        2.5287232398986816, 3.1440579891204834, 3.0381109714508057, 1.0
    ],
    'mean': [
        -0.20044206126280614, 0.9262122168593123, 0.6882957834577326,
        -0.34006432220172267, 0.06580817313563811, 0.003968959324292303,
        0.7713834946676629, 0.17298738585902546, 0.8228201278201596,
        0.619923833204487, -0.36062266594683073, -0.06273008083663979,
        0.0036269625481493103, 0.7859184988417455
    ],
    'std': [
        0.34305784707264697, 0.8880320790546445, 0.7302222416334363,
        0.6488505888815295, 0.29892353709320396, 0.5843509621147056,
        0.34319545241088767, 0.3199443608293499, 0.8737037120428734,
        0.7136347053099816, 0.6029010632268184, 0.28371581680727903,
        0.5371881924796513, 0.3407070455757959
    ],
    'q01': [
        -1.051365569829941, -3.5348887174584814e-14, 1.741881830823392e-16,
        -1.5984010362625123, -0.6003574305772781, -1.6678147149085998, 0.0,
        -0.4509398394823074, -2.5859282299029243e-14, 1.234041714549172e-16,
        -1.64055180311203, -1.2633746123313903, -1.7645285475254058, 0.0
    ],
    'q99': [
        0.5431358617544174, 2.495765209197998, 2.492974226474762,
        1.3241519677639007, 1.2496635341644287, 1.7392305016517635, 1.0,
        1.0814965963363647, 2.4167031812667843, 2.3470158338546754,
        1.141731116771698, 0.5208872479200363, 1.489715996980667, 1.0
    ],
    'count':
    1859602
}

_RELATIVE_ACTION_STATISTICS = {
    'min':
    [[
        -0.15623462200164795, -0.16764605045318604, -0.16970443725585938,
        -0.20000940561294556, -0.15340250730514526, -0.20000168681144714,
        -0.37838685512542725, -0.15502718091011047, -0.20030677318572998,
        -0.1721888780593872, -0.18998387455940247, -0.17636358737945557,
        -0.20000112056732178, -0.22537416219711304
    ],
     [
         -0.31191617250442505, -0.3346027135848999, -0.3386315107345581,
         -0.4000183045864105, -0.3058469295501709, -0.3976632058620453,
         -0.614814817905426, -0.3095979392528534, -0.38268542289733887,
         -0.3433429002761841, -0.3798891305923462, -0.35257601737976074,
         -0.3995015025138855, -0.3294653594493866
     ],
     [
         -0.4649693965911865, -0.5009051561355591, -0.50676429271698,
         -0.5988031029701233, -0.458637535572052, -0.5907891988754272,
         -0.614814817905426, -0.4638020694255829, -0.48982203006744385,
         -0.5141981840133667, -0.5652948617935181, -0.5271042585372925,
         -0.5919855833053589, -0.41561105847358704
     ],
     [
         -0.6162660121917725, -0.6651163101196289, -0.6723255515098572,
         -0.7199233174324036, -0.6113623380661011, -0.7699707746505737,
         -0.614814817905426, -0.6179124116897583, -0.6490402817726135,
         -0.6819474697113037, -0.7501269578933716, -0.7011628150939941,
         -0.776836633682251, -0.5020262002944946
     ],
     [
         -0.7658684253692627, -0.8280415534973145, -0.83699631690979,
         -0.8334474563598633, -0.7640075087547302, -0.9420039057731628,
         -0.614814817905426, -0.7617644667625427, -0.8069799542427063,
         -0.8491307497024536, -0.9262538552284241, -0.8718715906143188,
         -0.9684793949127197, -0.594243586063385
     ],
     [
         -0.9164783954620361, -0.9874361753463745, -0.9975383281707764,
         -1.00400972366333, -0.911322832107544, -1.1168217658996582,
         -0.7080556750297546, -0.8916434645652771, -0.9603351950645447,
         -1.011122703552246, -1.101320505142212, -1.0417617559432983,
         -1.1566747426986694, -0.6864609718322754
     ],
     [
         -1.0691542625427246, -1.1449098587036133, -1.1576461791992188,
         -1.1332952976226807, -1.051412582397461, -1.287815809249878,
         -0.7865169048309326, -0.9929096102714539, -1.1108429431915283,
         -1.1723419427871704, -1.2636971473693848, -1.2066856622695923,
         -1.3440144062042236, -0.7865169048309326
     ],
     [
         -1.2218137979507446, -1.2974035739898682, -1.3112123012542725,
         -1.2611026763916016, -1.1617908477783203, -1.4546704292297363,
         -0.898876428604126, -1.0969961881637573, -1.2596454620361328,
         -1.3262594938278198, -1.4245195388793945, -1.3703924417495728,
         -1.5263135433197021, -0.898876428604126
     ],
     [
         -1.372441053390503, -1.447338342666626, -1.4640965461730957,
         -1.4069215059280396, -1.2578787803649902, -1.6121726036071777, -1.0,
         -1.19899320602417, -1.4018747806549072, -1.479310154914856,
         -1.5692541599273682, -1.5275609493255615, -1.7078077793121338, -1.0
     ],
     [
         -1.5110177993774414, -1.5908403396606445, -1.608689546585083,
         -1.54514479637146, -1.3541831970214844, -1.7634245157241821, -1.0,
         -1.293934941291809, -1.5438759326934814, -1.6229406595230103,
         -1.7178936004638672, -1.683074951171875, -1.882439374923706, -1.0
     ],
     [
         -1.6355321407318115, -1.7311643362045288, -1.7522717714309692,
         -1.6814210414886475, -1.446195363998413, -1.9041634798049927, -1.0,
         -1.3676434755325317, -1.6762027740478516, -1.7657626867294312,
         -1.871291160583496, -1.8305044174194336, -2.056593179702759, -1.0
     ],
     [
         -1.7275934219360352, -1.8636142015457153, -1.885910153388977,
         -1.8326873779296875, -1.5357747077941895, -2.0358989238739014, -1.0,
         -1.4295730590820312, -1.808150053024292, -1.8981581926345825,
         -2.0156409740448, -1.9758203029632568, -2.221961736679077, -1.0
     ],
     [
         -1.8188360929489136, -1.992261290550232, -2.018099784851074,
         -1.9831218719482422, -1.6140577793121338, -2.1567585468292236, -1.0,
         -1.4801898002624512, -1.9286596775054932, -2.02775239944458,
         -2.157339572906494, -2.111522912979126, -2.387108564376831, -1.0
     ],
     [
         -1.8810598850250244, -2.1116244792938232, -2.1388092041015625,
         -2.1139721870422363, -1.6771425008773804, -2.265713930130005, -1.0,
         -1.5542949438095093, -2.0486526489257812, -2.147879123687744,
         -2.2884902954101562, -2.244635820388794, -2.542330026626587, -1.0
     ],
     [
         -1.936772346496582, -2.2265307903289795, -2.257511854171753,
         -2.2445836067199707, -1.740214467048645, -2.3640151023864746, -1.0,
         -1.6143319606781006, -2.1553502082824707, -2.261636734008789,
         -2.416691780090332, -2.3666346073150635, -2.6961402893066406, -1.0
     ],
     [
         -1.9831359386444092, -2.3307759761810303, -2.36329984664917,
         -2.3512368202209473, -1.7914525270462036, -2.4492745399475098, -1.0,
         -1.6693108081817627, -2.261469841003418, -2.36698317527771,
         -2.53256893157959, -2.4855473041534424, -2.839830160140991, -1.0
     ]],
    'max': [[
        0.15498490631580353, 0.20005442202091217, 0.20005667209625244,
        0.19985514879226685, 0.1528162956237793, 0.2001188099384308,
        0.7361559867858887, 0.15474265813827515, 0.1993194818496704,
        0.20014739036560059, 0.18120431900024414, 0.17337381839752197,
        0.20011717081069946, 0.4975297749042511
    ],
            [
                0.3098543882369995, 0.3882911205291748, 0.3993988037109375,
                0.37138789892196655, 0.30559873580932617, 0.3999616503715515,
                0.7361559867858887, 0.309272825717926, 0.3684011697769165,
                0.40029096603393555, 0.36051416397094727, 0.3459423780441284,
                0.39881736040115356, 0.4975297749042511
            ],
            [
                0.46472346782684326, 0.5428702235221863, 0.5929223299026489,
                0.5528702735900879, 0.4583788216114044, 0.5914298295974731,
                0.800000011920929, 0.4624479115009308, 0.5494232177734375,
                0.6004343032836914, 0.539147675037384, 0.5179338455200195,
                0.5909121632575989, 0.5006457567214966
            ],
            [
                0.6195164322853088, 0.7208306193351746, 0.7284860610961914,
                0.7331912517547607, 0.6110378503799438, 0.7748094797134399,
                0.800000011920929, 0.6139634847640991, 0.7279764413833618,
                0.8005720376968384, 0.7136830687522888, 0.6876122951507568,
                0.7716678380966187, 0.595525324344635
            ],
            [
                0.764305591583252, 0.8910004496574402, 0.9017477035522461,
                0.9056982398033142, 0.7636865973472595, 0.9420505166053772,
                0.800000011920929, 0.7645533084869385, 0.9014805555343628,
                1.0007017850875854, 0.8882356286048889, 0.855940580368042,
                0.9409557580947876, 0.6337910890579224
            ],
            [
                0.9164008498191833, 1.060915231704712, 1.0708266496658325,
                1.076215386390686, 0.9149380326271057, 1.1029270887374878,
                0.800000011920929, 0.9161303043365479, 1.0712734460830688,
                1.200827956199646, 1.0608360767364502, 1.020771861076355,
                1.0990257263183594, 0.7253433465957642
            ],
            [
                1.0687614679336548, 1.2189595699310303, 1.2328071594238281,
                1.2352322340011597, 1.04792320728302, 1.250866413116455,
                0.8189886808395386, 1.0687918663024902, 1.232945442199707,
                1.4009488821029663, 1.2299710512161255, 1.1832053661346436,
                1.2447912693023682, 0.8168955445289612
            ],
            [
                1.2193883657455444, 1.3759491443634033, 1.387863278388977,
                1.3918373584747314, 1.1360324621200562, 1.4218342304229736,
                0.9094943404197693, 1.2182191610336304, 1.3915128707885742,
                1.6010628938674927, 1.3963547945022583, 1.3427485227584839,
                1.3800389766693115, 0.908447802066803
            ],
            [
                1.355455756187439, 1.5183275938034058, 1.5341678857803345,
                1.5329556465148926, 1.2271716594696045, 1.5852620601654053,
                1.0, 1.3591952323913574, 1.5365383625030518,
                1.8011744022369385, 1.5581185817718506, 1.4983609914779663,
                1.5022748708724976, 1.0
            ],
            [
                1.4792304039001465, 1.6571649312973022, 1.6717768907546997,
                1.6718776226043701, 1.2969082593917847, 1.748584270477295, 1.0,
                1.438840627670288, 1.680770993232727, 2.001260757446289,
                1.7163355350494385, 1.6504261493682861, 1.637952208518982, 1.0
            ],
            [
                1.59755277633667, 1.780563235282898, 1.7978901863098145,
                1.7908905744552612, 1.3609228134155273, 1.9024004936218262,
                1.0, 1.5139979124069214, 1.8205636739730835,
                2.2013132572174072, 1.8687841892242432, 1.7970800399780273,
                1.7705798149108887, 1.0
            ],
            [
                1.6876590251922607, 1.897180199623108, 1.9143023490905762,
                1.9089956283569336, 1.4201512336730957, 2.056192398071289, 1.0,
                1.5542269945144653, 1.9613847732543945, 2.399487257003784,
                2.0168910026550293, 1.939436435699463, 1.8979809284210205, 1.0
            ],
            [
                1.7561901807785034, 1.9981712102890015, 2.017350435256958,
                2.0107533931732178, 1.481030821800232, 2.198575973510742, 1.0,
                1.5896430015563965, 2.101799964904785, 2.596306324005127,
                2.1580772399902344, 2.0752532482147217, 2.013698101043701, 1.0
            ],
            [
                1.8167648315429688, 2.0911967754364014, 2.1412651538848877,
                2.1323399543762207, 1.5314565896987915, 2.3408408164978027,
                1.0, 1.605363368988037, 2.2302868366241455, 2.7773444652557373,
                2.294142484664917, 2.206042528152466, 2.1297073364257812, 1.0
            ],
            [
                1.8602198362350464, 2.168449878692627, 2.2537410259246826,
                2.2472739219665527, 1.6166343688964844, 2.469943046569824, 1.0,
                1.6166977882385254, 2.3583338260650635, 2.952298402786255,
                2.4221291542053223, 2.329159736633301, 2.2449960708618164, 1.0
            ],
            [
                1.8887031078338623, 2.241629123687744, 2.3643198013305664,
                2.353733539581299, 1.684847116470337, 2.5986948013305664, 1.0,
                1.6447105407714844, 2.472832202911377, 3.1138815879821777,
                2.544224500656128, 2.4465250968933105, 2.3557143211364746, 1.0
            ]],
    'count':
    1807102
}

_ROBODOJO_STATS = {
    _STATISTIC_NAME: {
        'proprio': _STATE_STATISTICS,
        'action': _RELATIVE_ACTION_STATISTICS,
    },
}


def _split_statistics(flat_statistics, horizon_dependent=False):
    statistics = {}
    offset = 0
    for key, dim in _N17_LAYOUT:
        statistics[key] = {
            name: ([row[offset:offset + dim] for row in values]
                   if horizon_dependent else values[offset:offset + dim])
            for name, values in flat_statistics.items()
            if isinstance(values, list)
        }
        offset += dim
    return statistics


_N17_STATISTICS = {
    _EMBODIMENT_KEY:
    dict(
        state=_split_statistics(_STATE_STATISTICS),
        action=_split_statistics(_ABSOLUTE_ACTION_STATISTICS),
        relative_action=_split_statistics(
            _RELATIVE_ACTION_STATISTICS, horizon_dependent=True),
    ),
}
_N17_MODALITY_CONFIGS = {
    _EMBODIMENT_KEY:
    dict(
        video=dict(
            delta_indices=[0],
            modality_keys=['front', 'left_wrist', 'right_wrist'],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=[key for key, _ in _N17_LAYOUT],
        ),
        action=dict(
            delta_indices=list(range(_ACTION_WINDOW_SIZE)),
            modality_keys=[key for key, _ in _N17_LAYOUT],
            action_configs=[
                dict(
                    rep='RELATIVE',
                    type='NON_EEF',
                    format='DEFAULT',
                    state_key=None) for _ in _N17_LAYOUT
            ],
        ),
        language=dict(
            delta_indices=[0],
            modality_keys=['annotation.human.task_description']),
    ),
}

_ROBODOJO_GENERALIZATION_BASE_TASKS = ('stack_bowls', 'push_T',
                                       'pack_objects_into_box', 'fold_clothes',
                                       'hang_mugs', 'sweep_blocks',
                                       'pour_liquid_into_cup', 'make_toast',
                                       'arrange_largest_number',
                                       'sort_nesting_dolls_by_size',
                                       'store_laptop_and_headphones',
                                       'stack_blocks')
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

_PROCESSOR_KWARGS = dict(
    modality_configs=_N17_MODALITY_CONFIGS,
    statistics=_N17_STATISTICS,
    embodiment_id_mapping={_EMBODIMENT_KEY: _EMBODIMENT_ID},
    max_state_dim=132,
    max_action_dim=132,
    max_action_horizon=40,
    use_percentiles=True,
    clip_outliers=True,
    use_relative_action=True,
    apply_sincos_state_encoding=False,
    formalize_language=True,
    use_albumentations=True,
    shortest_image_edge=256,
    crop_fraction=0.95,
    image_target_size=(256, 256),
    image_crop_size=(230, 230),
    state_dropout_prob=0.2,
    color_jitter_params=dict(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
)

model = dict(
    embodiment_tag=_EMBODIMENT_KEY,
    processor_kwargs=dict(_delete_=True, **_PROCESSOR_KWARGS),
    use_relative_action=True,
)

# Restore the fine-tuned FluxVLA weights without reopening the source model.
inference_model = model.copy()
inference_model.update(
    model_path=None,
    load_metadata=False,
    load_pretrained_weights=False,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        _delete_=True,
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics=_ROBODOJO_STATS,
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_ROBODOJO_DATA_ROOT,
                statistic_name=_STATISTIC_NAME,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=1.0,
                repeat_to_full_length=False,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=_EMBODIMENT_ID,
                        parquet_keys=[
                            'observation.state',
                            'timestamp',
                            'actions',
                            'info',
                            'stats',
                            'action_masks',
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist',
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions'],
                        },
                    ),
                    dict(
                        type='RelativeActions',
                        mask=_RELATIVE_ACTION_MASK,
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_key='proprio',
                        action_key='action',
                        state_dim=132,
                        action_dim=132,
                        norm_type='quantile',
                        state_norm_type='quantile',
                        action_norm_type='min_max',
                        zero_constant_min_max_dims=True,
                        clip_norm=True,
                        normalization_epsilon=0.0,
                        preserve_input_dtype=True,
                    ),
                    dict(
                        type='PrepareStateActionTargets',
                        state_history_length=1,
                        action_horizon=40,
                        valid_action_dim=14,
                        state_dropout_prob=0.2,
                    ),
                    dict(
                        type='GrootN17ImageAugmentation',
                        embodiment_tag=_EMBODIMENT_KEY,
                        image_key='images',
                        output_image_key='images',
                        train_mode=True,
                        processor_kwargs=_PROCESSOR_KWARGS,
                    ),
                    dict(
                        type='QWen2VLImageTransform',
                        img_key='images',
                        size=dict(
                            shortest_edge=65536,
                            longest_edge=16777216,
                        ),
                        patch_size=16,
                        temporal_patch_size=2,
                        merge_size=2,
                        image_mean=[0.5, 0.5, 0.5],
                        image_std=[0.5, 0.5, 0.5],
                        to_tensor=True,
                    ),
                    dict(
                        type='ProcessPromptsWithImage',
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=_QWEN_TOKENIZER_PATH,
                            padding_side='left',
                            trust_remote_code=False,
                        ),
                        max_len=256,
                        add_system=False,
                        add_assistant_stub=False,
                        task_pos='after_images',
                        image_tag_template='',
                        img_start='<|vision_start|>',
                        img_end='<|vision_end|>',
                        img_context_token='<|image_pad|>',
                        img_tokens_source='from_image_grid_thw',
                        image_grid_thw_key='image_grid_thw',
                        image_merge_size=2,
                        padding_side='left',
                        use_eos_as_pad=False,
                        truncate=False,
                        lowercase_task_description=True,
                        strip_task_punctuation=True,
                        attention_mask_dtype='int64',
                        output_keys=[
                            'lang_tokens',
                            'lang_masks',
                            'images',
                            'image_grid_thw',
                            'states',
                            'actions',
                            'action_masks',
                            'embodiment_ids',
                            'sample_weight',
                        ],
                    ),
                ],
                action_window_size=_ACTION_WINDOW_SIZE,
                require_full_window=True,
                supervise_terminal_padding=False,
            ),
        ],
    ),
)

# 8 samples/GPU x total GPUs x 2 accumulation. Bound training by data
# passes so changing the machine count does not multiply exposure to data.
runner = dict(
    max_epochs=5,
    max_steps=None,
    save_epoch_interval=1,
    save_iter_interval=1000,
    max_keep_ckpts=10,
    enable_gradient_checkpointing=True,
    keep_params_fp32=True,
    metric=dict(active_trackers=('jsonl', 'wandb')),
)

eval = dict(
    _delete_=True,
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='groot_n17',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        state_dtype='fp32',
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                embodiment_id=_EMBODIMENT_ID,
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_key='proprio',
                action_key=None,
                state_dim=132,
                norm_type='quantile',
                clip_norm=True,
                normalization_epsilon=0.0,
                preserve_input_dtype=True,
                statistics_key='norm_stats',
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=40,
                valid_action_dim=14,
                state_dropout_prob=0.0,
            ),
            dict(
                type='GrootN17ImageAugmentation',
                embodiment_tag=_EMBODIMENT_KEY,
                image_key='pixel_values',
                output_image_key='pixel_values',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='QWen2VLImageTransform',
                img_key='pixel_values',
                size=dict(
                    shortest_edge=65536,
                    longest_edge=16777216,
                ),
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
                to_tensor=True,
            ),
            dict(
                type='ProcessPromptsWithImage',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_QWEN_TOKENIZER_PATH,
                    padding_side='left',
                    trust_remote_code=False,
                ),
                max_len=256,
                add_system=False,
                add_assistant_stub=False,
                task_pos='after_images',
                image_tag_template='',
                img_start='<|vision_start|>',
                img_end='<|vision_end|>',
                img_context_token='<|image_pad|>',
                img_tokens_source='from_image_grid_thw',
                image_grid_thw_key='image_grid_thw',
                image_merge_size=2,
                padding_side='left',
                use_eos_as_pad=False,
                truncate=False,
                lowercase_task_description=True,
                strip_task_punctuation=True,
                attention_mask_dtype='int64',
                output_keys=[
                    'lang_tokens',
                    'lang_masks',
                    'pixel_values',
                    'img_masks',
                    'image_grid_thw',
                    'states',
                    'embodiment_ids',
                    'replay_img',
                ],
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        norm_type='min_max',
        delta_action_mask=_RELATIVE_ACTION_MASK,
        clip_normalized_action=True,
        action_dim=14,
        statistic_name=_STATISTIC_NAME,
    ),
)

themis = dict(
    transport=dict(
        host='127.0.0.1',
        port=5555,
        timeout_s=30.0,
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key=_STATISTIC_NAME,
        image_encoding='rgb8',
        report_service_name='/fluxvla/report_evaluation',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False),
        model_client=dict(type='FluxVLAZMQModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=16,
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/gr00tn17_robodojo_relative_eval',
    ),
    ros_server=dict(
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/gr00tn17_robodojo_relative_eval'),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0,
        ),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context=dict(task_suite_name=_STATISTIC_NAME),
    ),
)
