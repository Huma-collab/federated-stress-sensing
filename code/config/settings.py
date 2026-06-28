from .utils import *

INFO = {"name": "Stress Recognition using Multimodal Sensory Dataset"}
NAME = 'ablation_no_imputation'
FUSION = "early"  # early, late,
DEVICE = select_device()
DATA = {}
CLASS = 2  # 2,3,5
USE_LAST_N_DAYS_DATA = 7
# cols division
HOURLY_HEADINGS = [
    "step",
    "act_in_vehicle",
    "act_on_bike",
    "act_still",
    "loc_dist",
    "loc_max_dis_from_campus",
    "loc_visit_num",
    "unlock_duration",
    "unlock_num",
]

DAILY = [
    "loc_food_dur",
    "loc_health_dur",
    "loc_home_dur",
    "loc_leisure_dur",
    "loc_other_dorm_dur",
    "loc_self_dorm_dur",
    "loc_social_dur",
    "loc_study_dur",
    "loc_workout_dur",
    "loc_worship_dur",
    "loc_food_still",
    "loc_home_still",
    "loc_other_dorm_still",
    "loc_self_dorm_still",
    "loc_social_still",
    "loc_study_still",
    "loc_food_unlock_duration",
    "loc_home_unlock_duration",
    "loc_other_dorm_unlock_duration",
    "loc_self_dorm_unlock_duration",
    "loc_social_unlock_duration",
    "loc_study_unlock_duration",
    "loc_food_unlock_num",
    "loc_home_unlock_num",
    "loc_other_dorm_unlock_num",
    "loc_self_dorm_unlock_num",
    "loc_social_unlock_num",
    "loc_study_unlock_num",
    "sleep_duration",
    "sleep_end",
    "sleep_start",
]

DIVISION = True
HEADINGS_HOURLY_ACT = ["step", "act_in_vehicle", "act_on_bike", "act_still"]
HEADINGS_HOURLY_LOC = ["loc_dist", "loc_max_dis_from_campus", "loc_visit_num"]
HEADINGS_HOURLY_UNLOCK = ["unlock_duration", "unlock_num"]

HEADINGS_DAILY_LOC_MOV = [
    "loc_food_dur",
    "loc_health_dur",
    "loc_home_dur",
    "loc_leisure_dur",
    "loc_other_dorm_dur",
    "loc_self_dorm_dur",
    "loc_social_dur",
    "loc_study_dur",
    "loc_workout_dur",
    "loc_worship_dur",
]
HEADINGS_DAILY_LOC_STILL = [
    "loc_food_still",
    "loc_home_still",
    "loc_other_dorm_still",
    "loc_self_dorm_still",
    "loc_social_still",
    "loc_study_still",
]
HEADINGS_DAILY_UNLOCK = [
    "loc_food_unlock_duration",
    "loc_home_unlock_duration",
    "loc_other_dorm_unlock_duration",
    "loc_self_dorm_unlock_duration",
    "loc_social_unlock_duration",
    "loc_study_unlock_duration",
    "loc_food_unlock_num",
    "loc_home_unlock_num",
    "loc_other_dorm_unlock_num",
    "loc_self_dorm_unlock_num",
    "loc_social_unlock_num",
    "loc_study_unlock_num",
]
HEADINGS_DAILY_SLEEP = ["sleep_duration", "sleep_end", "sleep_start"]

HEADINGS_HOURLY_ACT_HR = []
HEADINGS_HOURLY_LOC_HR = []
HEADINGS_HOURLY_UNLOCK_HR = []

for h in HEADINGS_HOURLY_ACT:
    for i in range(0, 24):
        HEADINGS_HOURLY_ACT_HR.append(f"{h}_hr_{i}")
for h in HEADINGS_HOURLY_LOC:
    for i in range(0, 24):
        HEADINGS_HOURLY_LOC_HR.append(f"{h}_hr_{i}")
for h in HEADINGS_HOURLY_UNLOCK:
    for i in range(0, 24):
        HEADINGS_HOURLY_UNLOCK_HR.append(f"{h}_hr_{i}")


TOTAL_HEADINGS = (
    len(HEADINGS_HOURLY_ACT)
    + len(HEADINGS_HOURLY_LOC)
    + len(HEADINGS_HOURLY_UNLOCK)
    + len(HEADINGS_DAILY_LOC_MOV)
    + len(HEADINGS_DAILY_LOC_STILL)
    + len(HEADINGS_DAILY_UNLOCK)
    + len(HEADINGS_DAILY_SLEEP)
)
ALL_HEADINGS_W_HR = (
    HEADINGS_HOURLY_ACT_HR
    + HEADINGS_HOURLY_LOC_HR
    + HEADINGS_HOURLY_UNLOCK_HR
    + HEADINGS_DAILY_LOC_MOV
    + HEADINGS_DAILY_LOC_STILL
    + HEADINGS_DAILY_UNLOCK
    + HEADINGS_DAILY_SLEEP
)
ALL_HEADINGS = (
    HEADINGS_HOURLY_ACT
    + HEADINGS_HOURLY_LOC
    + HEADINGS_HOURLY_UNLOCK
    + HEADINGS_DAILY_LOC_MOV
    + HEADINGS_DAILY_LOC_STILL
    + HEADINGS_DAILY_UNLOCK
    + HEADINGS_DAILY_SLEEP
)

HOURLY = False

# USERS = [ "1ff6d7f34acb354430e7323a35ff7703", "fa394f6d3d077bd5568fc3bc01580806", "596df3c263e97eb09d44a8535ec20ffa", "ea716dd032aaa0dcf8bfa36b1811917f", "3dad5f11680b432159b65838119ab87e", "a23be0935798553a76b4e74cfc1740f1", "ac70fe1f8115ac361f2023269c011c3e", "791c580346256bca5c697d19170a50b5", "2b3d267f18830c3c257b382edabef246", "46f1dfeda71a865074859c81bb314fd9", "f4d488e3c8096842b645280d8f01ae43", "3569e2f520db9014b4acc4227a6421c1", "3bb377ba0acb7d8916010184df36aa57", "8391d8b451084771d0affcc26e34d773", "79d8490ab94fc7cd6530b1f22593d708", "c31fd374e5d8227a31bf8f06f0402b96", "96cf40f9d3b2d69466161000cd69ae5c", "ab5ee451ce580a761fefd88f7b49a97f", "01fb41df0f6c2f69d65db5a38c600b4c", "c21283c11daebb61004b688f04c27715", "e60e86c5e68c34e34f69aa9257eb977b", "800994037be85bdfd6b3e43be70d02b3", "6683f5d8f1730b0fce6c1fd8b46ab76f", "0ab7acb36e6c710a7733c1c24e566bb3", "5580eba4ad4d1266854f3fec95266bd2", "003df5deff30e1e5a07b5d063fe85c3f", "284784fd7319c4e33b0bf1dc90c1a610", "2de8db6af975b72ac70440ede29e1cfb", "46b53cdf4d639d54e894d92b6dff817f", "46562035189535f838d640973dbb222e", "993b20b5d910d05b5d79e0de71386a49", "e06024d216d36ef2d7cddc8a67cf77a7", "c703d4732f997328085fad609ecd6656", "1c12f39481a6c09da17bd4e6bebed9a2", "9e9e21ee5dbdb37ffda3225cd0403d34", "59256b077a34f2f4ed7bbf7670e53786", "76b564dca3ab4d899e5a30d3cbcb0aac", "d2c8a48736cfe925440115558db860ee", "1fff60a0795f79ab1f9432d2cb1f56ee", "34ed21e64b190dc037d35ef7f646feb8", "a4dde9659b38b613f08b37c0c8bfa9eb", "07eab8e219b6a4dcd45167dc216e1189", "45555d3305e255895b11695c179a98cd", "f0db9e3e8b9b5080e5752383cceb34f6", "5fa34bfca33d0ced12ea991006fbc934", "f17129feb37d5dcec17728f50d7beb95", "8733d6d19519598608c6dc2f9ee525d7", "3d1a8b919f65dd22f661468f79b9b107", "87d88cba80cd67d705f61405fbe40da9", "c4de76cb7f30c4073617bfb80fabaec6", "f3e1995993cccfe83056a8deb2e315ff", "bd441b322153093e69f107a36a9b15d6", "13789d5f8a59f1e347e42b99233530d6", "cf007be9a060b779cf5a71b3e59a9d5b", "17fb1517cc9d7e416782bffb8edd9070", "e8811b53add5bfb32640af600ba33399", "2cc3f6274987ee627ab9be37181536eb", "1618e7f7049efc1a0fbe48af6fda8e57", "031cf9537e5da78c5a69a10cba088c94", "7e22b6d9cb821d7004409da02d95cb64", "ac02afb2367b89fdcab1b1245e195b27", "30b497738f121ad199afb509889f6cfe", "422875809be551a4ec179466aa8300ba", "da130c5ecc942e7cd705585222a42adf", "73e13f8273906f7f43a077f95ec48e7d", "b3db52ffec3adfc49eb061af2f2f2846", "690d25f3b083e647278ed85c6bb033ae", "5a254d648a95146b403690f81da84bc3", "db8df8be53722b041b1e649ae0c0b5d4", "3e7ca3545ad22d036db36c4f1e9f2e18", "8feafc5db9bf59d00a21335e69b57804", "91d30de367a6a0772fdca55ec5b129a2", "c0fec81b63a711ebc1c29a5df6e21b07", "c0b0998fe60a905081764378d1102494", "8ccf6026d9114437e881d73faa34ea51", "8c0c301d93ca93cae5c090caf21816ec", "db422d4921bbfae5352fdc90cea7bd07", "86a13f6d0bb7070ca042d60b2a582c4c", "2f706716305bceab3969e0f457da16b4", "703c70ff9f0c7a8e24757d0270a2e932", "bc3ae668f5930dcab54b63a9cfd0896d", "f5fa7e56d2303d8b1e3bb33d367c98c5", "e1b0f11a76278ee3f33e48e2d81b50d0", "165970c3847523de7691ef9766b0f3f4", "a4fcfc9e92d6861285d6c5979d2fd54b", "5e1d4d2718012c036ea65bfe3c80c0e7", "107c06248d28377b345ae06e5bda10c1", "6c4cf48742a55153dd574133a73a6932", "f40ed30e72159a5c839a43fe97e0c905", "600713d94c8ebe9f12f7d12c6479a131", "1df93e607e693903c6fce6054a0a67c5", "c2d80f95e15e8973a9a943564baea1b7", "e535002d3f0d00b98ff8a34754d6472b", "862f10a8c357e957a59d122077b3a5ad", "d1075cae00d6198eb840ad42a2df742c", "7a5f7cc91bafe10c9708342bb197ea58", "2d5d41b5e9a9a5bb7007264230e4c575", "84120765740b5395aa49a2feb12fbb43", "63e8142ef2fdd89aec0630ce6070831d", "0aee9faa6f2d2d24d85d2690f5b2df25", "1893420fc2b045e14ef0112c1b079afe", "56b90ef8a34e80ebc086ecd8d594e9d8", "b1fed5ab5fbd8a1b2fd93b925849a11c", "24e89780f6c9efef9ab28fea842cf620", "d1cd5ab675681cd980ee4dd212d8c53c", "98e5152e1020930fb007145c1c60648b", "546901e10be0157c4fed40c04bfeaaa5", "7dcfc929e033d32b768ff4b51bd6a748", "83e4705b8ead02d2659f4d4241183e17", "e946e1f85b5de2dba4edcf1aebbfdc0a", "480fca3c0b19afab1093cf2f01d63317", "968356427b2638af0930896c8d4a48d5", "0947cdc8e87a0a45a012aa387ccedbcd", "6ed487d275984c8c141d40ad10664f46", "23d00aa2ae3eb2c224734960f2391885", "8617ddac1f48b148e3683738519b2e7a", "ad15fc229da933fbf1fc0f92fc9b55a3", "eb5b94ef4672c2971d18f0d95d51f892", "4005a296ac7cd078b8bd56f0b905b7a6", "1e85c892d8f047ff621ad9134c4e6d8d", "fbcc8cd8254960ed44ffdd3390a2f6a0", "f663e0d08f34a40d5930a3298bc8e98e", "8ee53a46a77476104276e6a1af54472c", "1d2263527eed2a54e88d340fb8e55308", "61f9310e79c30ff4ff836fa065aee93a", "aeeb186fafcc356f44cae870555f4a0d", "b4d5a392109a5d0a0e6da33189e63511", "8c8ee0b0491acee56e6346c2af5c6cf4", "8730e74690f38a87355bf45b00fc220e", "64f2d26b3835e5fa744c35f1709e9c39", "f1dcebbb6fbe6d586ddb2c43c781e013", "83a37a7ffd416a2e0a863c86166a97c8", "20b8c420e5a7abcb07f11aab23e42923", "d060bfe5da35b9cbd05e3dc33e167092", "dfba04a156c13a94753a56bbbdd58969", "fc26652471ae91bdbbc11a8be89c362c", "fc73375dda5e5460f7088c78654a945e", "9ffbe25e279de21de86acd54b6fa60d0", "c75fc80fba7cbe2b7d9cbc71be5b91a6", "dd2ab53554046244203dfb4a72ad4e18", "8ee7d10d8b836222ee69ff04247971a4", "69d24b22972459bf83ab0c8eb24d13cc", "920e275acd8721dad396513f4d7c99ea", "6b22ac2851e90cebdb6bcc50288e031c", "4e37495c9c7c8a17e8737e2c61cfe228", "713a6a94988cc13bf0bb50ad0cb0a015", "8e12cde03a0eaf1a9c41596a81a1cf59", "6b4ad83b4ee50566e899f3b6c0fb7864", "62e43e3420646eaa7ecb44636abda31a", "a2949761d01e0f82cdbcc71231431959", "632503a1df16a201a0fc5a1b91d3b01c", "b5dea997cb5e8f17d428c5047a9a1c08", "1badfae62cc1b76787d4f8beb68737bf", "8c644d8cbd193c5575bb826a8abaa245", "964412e1937376da4b5a70412a600bc9", "8479552e5849c2f93e5f8ab1720ed78b", "c37f9221f44e9ca35a49180dc05a7587", "d86f28a1830ecec98f00cd0fddf8b1ef", "c98557ba920872193b0d6d8d90ac2eb0", "7d2c632a05bbb03ca97555d61be83c41", "12e0a85ee9819ae1abeca0ccfaa52d13", "f578948fdf09c92bc603dae7400b6673", "aa77ecb82c267ad01e9899413f072025", "2c4f43b2212eee5ba69563f139911138", "1741432fb469dbbcd5a40599670c64ce", "51ac1dd0849798d75614176ebff633be", "4eeb9f06b3ff630aa8f81fd23e1dfe7b", "ffc4b142e017c162ed4db7b05414fc4b", "e3d27e46b8e06963920efb69f95df6fc", "bdefb86f43d463ae4475291ec837f7d7", "212521733ea8eeff63d997dc0213c692", "35cf1abf179310dc33907d953f590366", "ac23ab17a31d1abeea72e408ac5871c1", "9a67ed145c6e7be55b1f744897316f71", "0b9c2303bcc61063f9a5653d77c90501", "bbcad50219ee8a1484b4c6f000eb4e68", "4b5eb55e64b6bc941796bf114149cab7", "a6c1b8b62cc50f9022f018b5198648a0", "62caae288910745b5eb9dde5a9873179", "59cf760cd019690c9007452187578915", "c21d2515be8477a92181b9c0035a9884", "a7cb6ba9ef6b3878a4284b0d63afecd7", "3528757eaf1d15ee6c778eb79cd02de7", "0107c61e54459068bb83f6be2058d65d", "a7a320d21e232075512a494c982dc17f", "4a8ac645226348425b0a43a8373e394d", "c7d47e96f38254e31508ca2c19b24d29", "2d06da49889845dc3af964bafb630b0a", "557b44c8daf2ddb5d02313f20f2505b9", "0f2be96bde481cd3898f147a995a7d56", "bd6de07de02a8c2e98018a1c7daecc87", "0f727c1018c74fbb7142e8c7a77840d6", "429cf5f822600da91a6f4227c0af86c6", "d038cb570b5de5840c0677e72d2ba198", "0ba15aa0582c5e825710d42fe3eb231d", "b855dfddb71e240c1990b539533cd1f6", "f066d6d9013a60a4b65eda0f2518f61a", "54446e36f4601e2c4ab70c355baccc32", "1d0fa4dfa0545bedeaa6045c70b6d19c", "bea4c7250c8631ddfacf57dfa386caba", "12f90c19f2c2b74ec542b62e74e7cad1", "6294895f1093b3878b385812d65c3197", "f7472689b3556548c03d3b064d6ab9a8", "58863d024fc04ff1c3082e2c81e14cf2", "7d41e1ac6994aa87a10a923b31b16cd8", "50e32d5c464c892af7aa4b6a666e876d", "169d3212668c4eed434dafd32e33c946", "97487f1632fbab90426f6760a0fe94ea", "b17eeda399b312140dc40b8445a8e0a5", "5e8e425fb65143d25c73b88bd26638eb", "84875d7ec9c81d58d687aa4805c35c77", "5c446b55e04c641913c09fd39d01d32b", "f5529fbae87a8d170937d3e39d5a63cc", "374d8bd8b755b55ed5a9b08304db6cee", "03a0ce5623bfeb8aa3113605f7682215", "fe8ddda4ae8c71f7054ca024b82f5c98", "a52b5e80b4c7a8e05f8cc0a16ae4ea9f","6b0083d00297f9c03e00b2cde889b666"]
# USERS =  ['003df5deff30e1e5a07b5d063fe85c3f', '0107c61e54459068bb83f6be2058d65d', '03a0ce5623bfeb8aa3113605f7682215', '0947cdc8e87a0a45a012aa387ccedbcd', '0ab7acb36e6c710a7733c1c24e566bb3', '0f2be96bde481cd3898f147a995a7d56', '0f727c1018c74fbb7142e8c7a77840d6', '107c06248d28377b345ae06e5bda10c1', '12f90c19f2c2b74ec542b62e74e7cad1', '13789d5f8a59f1e347e42b99233530d6', '1618e7f7049efc1a0fbe48af6fda8e57', '1741432fb469dbbcd5a40599670c64ce', '17fb1517cc9d7e416782bffb8edd9070', '1d0fa4dfa0545bedeaa6045c70b6d19c', '1df93e607e693903c6fce6054a0a67c5', '1ff6d7f34acb354430e7323a35ff7703', '20b8c420e5a7abcb07f11aab23e42923', '212521733ea8eeff63d997dc0213c692', '23d00aa2ae3eb2c224734960f2391885', '284784fd7319c4e33b0bf1dc90c1a610', '2b3d267f18830c3c257b382edabef246', '2cc3f6274987ee627ab9be37181536eb', '2d06da49889845dc3af964bafb630b0a', '2d5d41b5e9a9a5bb7007264230e4c575', '2de8db6af975b72ac70440ede29e1cfb', '2f706716305bceab3969e0f457da16b4', '30b497738f121ad199afb509889f6cfe', '34ed21e64b190dc037d35ef7f646feb8', '3569e2f520db9014b4acc4227a6421c1', '3bb377ba0acb7d8916010184df36aa57', '3dad5f11680b432159b65838119ab87e', '4005a296ac7cd078b8bd56f0b905b7a6', '422875809be551a4ec179466aa8300ba', '45555d3305e255895b11695c179a98cd', '46562035189535f838d640973dbb222e', '46b53cdf4d639d54e894d92b6dff817f', '480fca3c0b19afab1093cf2f01d63317', '4eeb9f06b3ff630aa8f81fd23e1dfe7b', '51ac1dd0849798d75614176ebff633be', '546901e10be0157c4fed40c04bfeaaa5', '557b44c8daf2ddb5d02313f20f2505b9', '5580eba4ad4d1266854f3fec95266bd2', '58863d024fc04ff1c3082e2c81e14cf2', '5a254d648a95146b403690f81da84bc3', '5c446b55e04c641913c09fd39d01d32b', '5fa34bfca33d0ced12ea991006fbc934', '600713d94c8ebe9f12f7d12c6479a131', '61f9310e79c30ff4ff836fa065aee93a', '62e43e3420646eaa7ecb44636abda31a', '63e8142ef2fdd89aec0630ce6070831d', '64f2d26b3835e5fa744c35f1709e9c39', '6683f5d8f1730b0fce6c1fd8b46ab76f', '6ed487d275984c8c141d40ad10664f46', '713a6a94988cc13bf0bb50ad0cb0a015', '791c580346256bca5c697d19170a50b5', '79d8490ab94fc7cd6530b1f22593d708', '7a5f7cc91bafe10c9708342bb197ea58', '7d41e1ac6994aa87a10a923b31b16cd8', '7dcfc929e033d32b768ff4b51bd6a748', '7e22b6d9cb821d7004409da02d95cb64', '800994037be85bdfd6b3e43be70d02b3', '8391d8b451084771d0affcc26e34d773', '83a37a7ffd416a2e0a863c86166a97c8', '83e4705b8ead02d2659f4d4241183e17', '84120765740b5395aa49a2feb12fbb43', '8617ddac1f48b148e3683738519b2e7a', '86a13f6d0bb7070ca042d60b2a582c4c', '8730e74690f38a87355bf45b00fc220e', '87d88cba80cd67d705f61405fbe40da9', '8c0c301d93ca93cae5c090caf21816ec', '8c644d8cbd193c5575bb826a8abaa245', '8c8ee0b0491acee56e6346c2af5c6cf4', '8ccf6026d9114437e881d73faa34ea51', '8e12cde03a0eaf1a9c41596a81a1cf59', '920e275acd8721dad396513f4d7c99ea', '964412e1937376da4b5a70412a600bc9', '968356427b2638af0930896c8d4a48d5', '96cf40f9d3b2d69466161000cd69ae5c', '97487f1632fbab90426f6760a0fe94ea', '993b20b5d910d05b5d79e0de71386a49', '9a67ed145c6e7be55b1f744897316f71', 'a23be0935798553a76b4e74cfc1740f1', 'a2949761d01e0f82cdbcc71231431959', 'a4fcfc9e92d6861285d6c5979d2fd54b', 'a7a320d21e232075512a494c982dc17f', 'a7cb6ba9ef6b3878a4284b0d63afecd7', 'aa77ecb82c267ad01e9899413f072025', 'ac02afb2367b89fdcab1b1245e195b27', 'ac23ab17a31d1abeea72e408ac5871c1', 'b17eeda399b312140dc40b8445a8e0a5', 'b1fed5ab5fbd8a1b2fd93b925849a11c', 'b3db52ffec3adfc49eb061af2f2f2846', 'b5dea997cb5e8f17d428c5047a9a1c08', 'b855dfddb71e240c1990b539533cd1f6', 'bc3ae668f5930dcab54b63a9cfd0896d', 'bdefb86f43d463ae4475291ec837f7d7', 'bea4c7250c8631ddfacf57dfa386caba', 'c0fec81b63a711ebc1c29a5df6e21b07', 'c21283c11daebb61004b688f04c27715', 'c2d80f95e15e8973a9a943564baea1b7', 'c31fd374e5d8227a31bf8f06f0402b96', 'c703d4732f997328085fad609ecd6656', 'c75fc80fba7cbe2b7d9cbc71be5b91a6', 'cf007be9a060b779cf5a71b3e59a9d5b', 'd038cb570b5de5840c0677e72d2ba198', 'd060bfe5da35b9cbd05e3dc33e167092', 'd1075cae00d6198eb840ad42a2df742c', 'd1cd5ab675681cd980ee4dd212d8c53c', 'd2c8a48736cfe925440115558db860ee', 'd86f28a1830ecec98f00cd0fddf8b1ef', 'dfba04a156c13a94753a56bbbdd58969', 'e06024d216d36ef2d7cddc8a67cf77a7', 'e1b0f11a76278ee3f33e48e2d81b50d0', 'e3d27e46b8e06963920efb69f95df6fc', 'e535002d3f0d00b98ff8a34754d6472b', 'e60e86c5e68c34e34f69aa9257eb977b', 'e8811b53add5bfb32640af600ba33399', 'e946e1f85b5de2dba4edcf1aebbfdc0a', 'eb5b94ef4672c2971d18f0d95d51f892', 'f066d6d9013a60a4b65eda0f2518f61a', 'f0db9e3e8b9b5080e5752383cceb34f6', 'f17129feb37d5dcec17728f50d7beb95', 'f3e1995993cccfe83056a8deb2e315ff', 'f4d488e3c8096842b645280d8f01ae43', 'f5529fbae87a8d170937d3e39d5a63cc', 'f578948fdf09c92bc603dae7400b6673', 'f663e0d08f34a40d5930a3298bc8e98e', 'fa394f6d3d077bd5568fc3bc01580806', 'fc73375dda5e5460f7088c78654a945e', 'fe8ddda4ae8c71f7054ca024b82f5c98']
USERS = [
    "03a0ce5623bfeb8aa3113605f7682215",
    "0947cdc8e87a0a45a012aa387ccedbcd",
    "0ab7acb36e6c710a7733c1c24e566bb3",
    # "0f2be96bde481cd3898f147a995a7d56",
    "0f727c1018c74fbb7142e8c7a77840d6",
    "107c06248d28377b345ae06e5bda10c1",
    "12f90c19f2c2b74ec542b62e74e7cad1",
    "13789d5f8a59f1e347e42b99233530d6",
    # "1618e7f7049efc1a0fbe48af6fda8e57",
    "1741432fb469dbbcd5a40599670c64ce",
    "17fb1517cc9d7e416782bffb8edd9070",
    # "1d0fa4dfa0545bedeaa6045c70b6d19c",
    "1df93e607e693903c6fce6054a0a67c5",
    "1ff6d7f34acb354430e7323a35ff7703",
    "20b8c420e5a7abcb07f11aab23e42923",
    "212521733ea8eeff63d997dc0213c692",
    "23d00aa2ae3eb2c224734960f2391885",
    "2b3d267f18830c3c257b382edabef246",
    "2cc3f6274987ee627ab9be37181536eb",
    "2d06da49889845dc3af964bafb630b0a",
    "2d5d41b5e9a9a5bb7007264230e4c575",
    "2de8db6af975b72ac70440ede29e1cfb",
    # "2f706716305bceab3969e0f457da16b4",
    "34ed21e64b190dc037d35ef7f646feb8",
    "3569e2f520db9014b4acc4227a6421c1",
    "3dad5f11680b432159b65838119ab87e",
    # "4005a296ac7cd078b8bd56f0b905b7a6",
    "422875809be551a4ec179466aa8300ba",
    "45555d3305e255895b11695c179a98cd",
    "480fca3c0b19afab1093cf2f01d63317",
    "51ac1dd0849798d75614176ebff633be",
    "546901e10be0157c4fed40c04bfeaaa5",
    "5580eba4ad4d1266854f3fec95266bd2",
    "5c446b55e04c641913c09fd39d01d32b",
    "5fa34bfca33d0ced12ea991006fbc934",
    "600713d94c8ebe9f12f7d12c6479a131",
    # "62e43e3420646eaa7ecb44636abda31a",
    "63e8142ef2fdd89aec0630ce6070831d",
    "64f2d26b3835e5fa744c35f1709e9c39",
    # "6683f5d8f1730b0fce6c1fd8b46ab76f",
    "6ed487d275984c8c141d40ad10664f46",
    "713a6a94988cc13bf0bb50ad0cb0a015",
    "791c580346256bca5c697d19170a50b5",
    "79d8490ab94fc7cd6530b1f22593d708",
    "7d41e1ac6994aa87a10a923b31b16cd8",
    "7dcfc929e033d32b768ff4b51bd6a748",
    "800994037be85bdfd6b3e43be70d02b3",
    "8391d8b451084771d0affcc26e34d773",
    "83a37a7ffd416a2e0a863c86166a97c8",
    "83e4705b8ead02d2659f4d4241183e17",
    "84120765740b5395aa49a2feb12fbb43",
    "8617ddac1f48b148e3683738519b2e7a",
    "86a13f6d0bb7070ca042d60b2a582c4c",
    "8730e74690f38a87355bf45b00fc220e",
    "8c0c301d93ca93cae5c090caf21816ec",
    # "8c644d8cbd193c5575bb826a8abaa245",
    # "8c8ee0b0491acee56e6346c2af5c6cf4",
    "8ccf6026d9114437e881d73faa34ea51",
    # "8e12cde03a0eaf1a9c41596a81a1cf59",
    "920e275acd8721dad396513f4d7c99ea",
    "964412e1937376da4b5a70412a600bc9",
    # "968356427b2638af0930896c8d4a48d5",
    "96cf40f9d3b2d69466161000cd69ae5c",
    "97487f1632fbab90426f6760a0fe94ea",
    "993b20b5d910d05b5d79e0de71386a49",
    "a23be0935798553a76b4e74cfc1740f1",
    # "a2949761d01e0f82cdbcc71231431959",
    # "a4fcfc9e92d6861285d6c5979d2fd54b",
    "ac02afb2367b89fdcab1b1245e195b27",
    "ac23ab17a31d1abeea72e408ac5871c1",
    "b17eeda399b312140dc40b8445a8e0a5",
    "b1fed5ab5fbd8a1b2fd93b925849a11c",
    "b3db52ffec3adfc49eb061af2f2f2846",
    "b5dea997cb5e8f17d428c5047a9a1c08",
    "b855dfddb71e240c1990b539533cd1f6",
    "bc3ae668f5930dcab54b63a9cfd0896d",
    "bdefb86f43d463ae4475291ec837f7d7",
    # "bea4c7250c8631ddfacf57dfa386caba",
    "c0fec81b63a711ebc1c29a5df6e21b07",
    "c21283c11daebb61004b688f04c27715",
    "c2d80f95e15e8973a9a943564baea1b7",
    "c703d4732f997328085fad609ecd6656",
    "c75fc80fba7cbe2b7d9cbc71be5b91a6",
    "cf007be9a060b779cf5a71b3e59a9d5b",
    "d038cb570b5de5840c0677e72d2ba198",
    "d060bfe5da35b9cbd05e3dc33e167092",
    "d1075cae00d6198eb840ad42a2df742c",
    "d1cd5ab675681cd980ee4dd212d8c53c",
    "d2c8a48736cfe925440115558db860ee",
    "e06024d216d36ef2d7cddc8a67cf77a7",
    "e1b0f11a76278ee3f33e48e2d81b50d0",
    "e3d27e46b8e06963920efb69f95df6fc",
    # "e535002d3f0d00b98ff8a34754d6472b",
    "e946e1f85b5de2dba4edcf1aebbfdc0a",
    # "f066d6d9013a60a4b65eda0f2518f61a",
    "f0db9e3e8b9b5080e5752383cceb34f6",
    "f17129feb37d5dcec17728f50d7beb95",
    "f3e1995993cccfe83056a8deb2e315ff",
    "f5529fbae87a8d170937d3e39d5a63cc",
    "f578948fdf09c92bc603dae7400b6673",
    # "f663e0d08f34a40d5930a3298bc8e98e",
    "fa394f6d3d077bd5568fc3bc01580806",
    "fc73375dda5e5460f7088c78654a945e",
    "fe8ddda4ae8c71f7054ca024b82f5c98",
]


if HOURLY:
    FEATURES = []
    for h in HOURLY_HEADINGS:
        for i in range(0, 24):
            FEATURES.append(f"{h}_hr_{i}")
else:
    FEATURES = DAILY
if DIVISION:
    FEATURES = ALL_HEADINGS
SELECTED_USERS = USERS
BATCH_SIZE = 20
# hyperparameters
LSTM_CLASSIFIER = {
    "learning_rate": 0.0001,
    "log_location": "",
    "input_size": (len(HOURLY_HEADINGS) if HOURLY else len(FEATURES)),
    "hidden_size": 64,
    "latent_size": 32,
    "save_location": "saved_models/lstm_classifier",
}


PATIENCE = 50
MIN_DELTA = 0.001


# FEDERATED SETTINGS
# EPOCHS = 50000
# GLOBAL_EPOCHS = 10
# LOG_LOCATION = "logs"
# CLIENTS_NUM = len(USERS)

EPOCHS = 200
GLOBAL_EPOCHS = 5
LOG_LOCATION = "logs"
CLIENTS_NUM = len(USERS)
RANDOM_CLIENTS = False
CLIENTS = []
if not RANDOM_CLIENTS:
    CLIENTS = USERS[0:CLIENTS_NUM]

# Ablation model selection: lstm, lstm_3ch, lstm_4ch

# Differential Privacy settings
DP_ENABLED = True          # Toggle DP on/off
DP_NOISE_MULTIPLIER = 0.1   # Gaussian noise scale (sigma) — higher = more privacy, less accuracy
DP_CLIPPING_THRESHOLD = 1.0 # Max L2 norm for weight clipping (sensitivity)
MODEL_NAME = 'lstm'
