from read_pdb_file import *
import os
import numpy as np
from tqdm import tqdm_notebook as tqdm


pro_list = ['2023']
lig_list = ['2003', '2006', '2009', '2010', '2012', '2013', '2014', '2015', '2017', '2018', '2019', '2020', '2023', '2025', '2027', '2034', '2038', '2039', '2041', '2042', '2047', '2048', '2049', '2050', '2052', '2053', '2054', '2055', '2059', '2060', '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2072', '2073', '2078', '2079', '2085', '2089', '2091', '2092', '2093', '2094', '2096', '2098', '2100', '2102', '2103', '2105', '2106', '2107', '2108', '2119', '2123', '2125', '2126', '2127', '2128', '2129', '2131', '2132', '2134', '2135', '2138', '2139', '2141', '2144', '2145', '2148', '2149', '2152', '2153', '2154', '2155', '2156', '2159', '2162', '2164', '2165', '2168', '2169', '2170', '2174', '2176', '2184', '2186', '2187', '2188', '2190', '2191', '2193', '2194', '2195', '2196', '2198', '2200', '2201', '2202', '2203', '2204', '2207', '2209', '2210', '2211', '2215', '2217', '2223', '2233', '2234', '2235', '2236', '2240', '2241', '2242', '2243', '2244', '2245', '2247', '2250', '2252', '2254', '2256', '2258', '2262', '2263', '2266', '2267', '2268', '2269', '2270', '2272', '2275', '2276', '2279', '2285', '2288', '2290', '2293', '2299', '2300', '2301', '2303', '2305', '2306', '2311', '2312', '2313', '2314', '2316', '2317', '2320', '2323', '2324', '2326', '2328', '2330', '2331', '2333', '2336', '2337', '2338', '2339', '2340', '2341', '2343', '2344', '2346', '2351', '2354', '2356', '2357', '2359', '2360', '2361', '2362', '2364', '2365', '2368', '2369', '2370', '2371', '2372', '2375', '2376', '2378', '2379', '2382', '2384', '2385', '2386', '2388', '2389', '2394', '2395', '2399', '2401', '2402', '2403', '2405', '2406', '2409', '2411', '2412', '2413', '2415', '2418', '2420', '2423', '2424', '2425', '2426', '2429', '2430', '2431', '2434', '2436', '2437', '2439', '2441', '2443', '2444', '2445', '2446', '2448', '2450', '2453', '2454', '2455', '2456', '2461', '2463', '2465', '2470', '2472', '2475', '2476', '2478', '2479', '2482', '2484', '2487', '2488', '2489', '2491', '2495', '2496', '2497', '2499', '2500', '2501', '2503', '2505', '2506', '2507', '2508', '2511', '2512', '2513', '2519', '2520', '2521', '2522', '2523', '2527', '2529', '2531', '2532', '2536', '2538', '2540', '2541', '2542', '2543', '2545', '2548', '2549', '2550', '2551', '2554', '2557', '2558', '2560', '2562', '2563', '2564', '2566', '2567', '2573', '2574', '2575', '2576', '2580', '2581', '2582', '2583', '2584', '2588', '2589', '2591', '2597', '2598', '2599', '2600', '2602', '2607', '2610', '2612', '2613', '2617', '2618', '2619', '2621', '2624', '2625', '2629', '2634', '2636', '2637', '2640', '2641', '2642', '2644', '2645', '2649', '2652', '2655', '2658', '2660', '2662', '2663', '2667', '2668', '2669', '2670', '2671', '2672', '2673', '2674', '2675', '2677', '2679', '2680', '2682', '2683', '2685', '2687', '2690', '2691', '2692', '2693', '2696', '2697', '2698', '2701', '2705', '2707', '2708', '2709', '2710', '2711', '2715', '2717', '2718', '2722', '2725', '2726', '2727', '2729', '2732', '2733', '2734', '2735', '2736', '2740', '2744', '2746', '2747', '2749', '2751', '2752', '2753', '2755', '2759', '2760', '2763', '2764', '2767', '2768', '2772', '2774', '2776', '2781', '2784', '2786', '2792', '2794', '2796', '2799', '2801', '2805', '2813', '2815', '2816', '2817', '2819', '2822', '2823', '2827', '2828', '2831', '2834', '2836', '2837', '2838', '2841', '2842', '2843', '2844', '2846', '2850', '2852', '2854', '2857', '2858', '2859', '2860', '2862', '2863', '2864', '2865', '2867', '2869', '2874', '2876', '2878', '2879', '2882', '2883', '2887', '2891', '2892', '2894', '2895', '2896', '2897', '2898', '2900', '2901', '2902', '2904', '2906', '2907', '2908', '2909', '2912', '2913', '2915', '2918', '2919', '2921', '2923', '2924', '2925', '2927', '2929', '2933', '2935', '2937', '2939', '2941', '2942', '2943', '2944', '2945', '2946', '2948', '2949', '2950', '2951', '2952', '2953', '2957', '2959', '2962', '2963', '2964', '2965', '2966', '2968', '2970', '2972', '2975', '2977', '2980', '2981', '2982', '2983', '2985', '2987', '2988', '2991', '2995', '2999', '3000']

# for i in range(len(pro)):
#     for j in range(len(lig)):
#         x, y = pro_lig_reader_sample(pro_label=pro[i], lig_label=lig[j],folder_name='test_data')
#         plot_3D(x)

# 2023: ['2003', '2006', '2009', '2010', '2012', '2013', '2014', '2015', '2017', '2018', '2019', '2020', '2023', '2025', '2027', '2034', '2038', '2039', '2041', '2042', '2047', '2048', '2049', '2050', '2052', '2053', '2054', '2055', '2059', '2060', '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2072', '2073', '2078', '2079', '2085', '2089', '2091', '2092', '2093', '2094', '2096', '2098', '2100', '2102', '2103', '2105', '2106', '2107', '2108', '2119', '2123', '2125', '2126', '2127', '2128', '2129', '2131', '2132', '2134', '2135', '2138', '2139', '2141', '2144', '2145', '2148', '2149', '2152', '2153', '2154', '2155', '2156', '2159', '2162', '2164', '2165', '2168', '2169', '2170', '2174', '2176', '2184', '2186', '2187', '2188', '2190', '2191', '2193', '2194', '2195', '2196', '2198', '2200', '2201', '2202', '2203', '2204', '2207', '2209', '2210', '2211', '2215', '2217', '2223', '2233', '2234', '2235', '2236', '2240', '2241', '2242', '2243', '2244', '2245', '2247', '2250', '2252', '2254', '2256', '2258', '2262', '2263', '2266', '2267', '2268', '2269', '2270', '2272', '2275', '2276', '2279', '2285', '2288', '2290', '2293', '2299', '2300', '2301', '2303', '2305', '2306', '2311', '2312', '2313', '2314', '2316', '2317', '2320', '2323', '2324', '2326', '2328', '2330', '2331', '2333', '2336', '2337', '2338', '2339', '2340', '2341', '2343', '2344', '2346', '2351', '2354', '2356', '2357', '2359', '2360', '2361', '2362', '2364', '2365', '2368', '2369', '2370', '2371', '2372', '2375', '2376', '2378', '2379', '2382', '2384', '2385', '2386', '2388', '2389', '2394', '2395', '2399', '2401', '2402', '2403', '2405', '2406', '2409', '2411', '2412', '2413', '2415', '2418', '2420', '2423', '2424', '2425', '2426', '2429', '2430', '2431', '2434', '2436', '2437', '2439', '2441', '2443', '2444', '2445', '2446', '2448', '2450', '2453', '2454', '2455', '2456', '2461', '2463', '2465', '2470', '2472', '2475', '2476', '2478', '2479', '2482', '2484', '2487', '2488', '2489', '2491', '2495', '2496', '2497', '2499', '2500', '2501', '2503', '2505', '2506', '2507', '2508', '2511', '2512', '2513', '2519', '2520', '2521', '2522', '2523', '2527', '2529', '2531', '2532', '2536', '2538', '2540', '2541', '2542', '2543', '2545', '2548', '2549', '2550', '2551', '2554', '2557', '2558', '2560', '2562', '2563', '2564', '2566', '2567', '2573', '2574', '2575', '2576', '2580', '2581', '2582', '2583', '2584', '2588', '2589', '2591', '2597', '2598', '2599', '2600', '2602', '2607', '2610', '2612', '2613', '2617', '2618', '2619', '2621', '2624', '2625', '2629', '2634', '2636', '2637', '2640', '2641', '2642', '2644', '2645', '2649', '2652', '2655', '2658', '2660', '2662', '2663', '2667', '2668', '2669', '2670', '2671', '2672', '2673', '2674', '2675', '2677', '2679', '2680', '2682', '2683', '2685', '2687', '2690', '2691', '2692', '2693', '2696', '2697', '2698', '2701', '2705', '2707', '2708', '2709', '2710', '2711', '2715', '2717', '2718', '2722', '2725', '2726', '2727', '2729', '2732', '2733', '2734', '2735', '2736', '2740', '2744', '2746', '2747', '2749', '2751', '2752', '2753', '2755', '2759', '2760', '2763', '2764', '2767', '2768', '2772', '2774', '2776', '2781', '2784', '2786', '2792', '2794', '2796', '2799', '2801', '2805', '2813', '2815', '2816', '2817', '2819', '2822', '2823', '2827', '2828', '2831', '2834', '2836', '2837', '2838', '2841', '2842', '2843', '2844', '2846', '2850', '2852', '2854', '2857', '2858', '2859', '2860', '2862', '2863', '2864', '2865', '2867', '2869', '2874', '2876', '2878', '2879', '2882', '2883', '2887', '2891', '2892', '2894', '2895', '2896', '2897', '2898', '2900', '2901', '2902', '2904', '2906', '2907', '2908', '2909', '2912', '2913', '2915', '2918', '2919', '2921', '2923', '2924', '2925', '2927', '2929', '2933', '2935', '2937', '2939', '2941', '2942', '2943', '2944', '2945', '2946', '2948', '2949', '2950', '2951', '2952', '2953', '2957', '2959', '2962', '2963', '2964', '2965', '2966', '2968', '2970', '2972', '2975', '2977', '2980', '2981', '2982', '2983', '2985', '2987', '2988', '2991', '2995', '2999', '3000']
#        [1.203532716630504, 6.083967208984611, 2.580146507468131, 1.3930100502150007, 0.3548365821050587, 6.107975196413293, 5.422573282123536, 6.495533927245704, 1.2015435905534186, 1.9622369887452427, 1.9771337840419403, 4.00129903906219, 3.7535170174118053, 1.6801958219207636, 1.7602891807882042, 1.1469442009095292, 0.7413035815372784, 5.139041155702103, 3.593262306038899, 1.2473640206451366, 5.461197029955976, 0.9776343897388238, 1.0084131097918148, 3.60017444021814, 2.130349032435765, 3.5558249394479478, 5.28456554505666, 1.8441193019975666, 2.318071180960584, 5.247236034332742, 2.4068859549218358, 4.307351274275181, 1.2171228368574738, 1.3269698564775285, 1.872209657063013, 2.2340915379634736, 0.8286247642932237, 1.5732110475076118, 1.037314802747939, 3.0627486021546075, 1.9938761245373295, 1.0146161835886496, 1.974637941497125, 3.363010704710885, 1.2085309263730044, 1.5299692807373646, 1.0077206954310292, 2.2678760989084035, 1.9485425322532697, 1.957883806562585, 1.0843329746899655, 2.631958586300325, 1.451595329284301, 1.1727275898519633, 0.8402797153329378, 1.1129371051411685, 5.569926480663816, 2.1582889055916503, 1.8017907758671652, 1.9935079633650874, 1.7684877720809984, 1.9875686151677863, 2.2230955894877775, 2.7896691560111573, 1.3041299781846882, 1.5505218476371125, 1.1760297615281692, 2.014808675780407, 1.6087911610895929, 1.6985314244958796, 0.8393574923713967, 5.993857188822569, 1.7674405223373155, 1.7387472501775523, 6.835380091845659, 1.2066267028372937, 2.962323581244969, 2.106656355459999, 0.6872183059261454, 2.0627530147838833, 3.571955906782726, 5.9659528995794116, 1.5245461619773915, 2.6235125309401517, 3.0488569989423935, 3.2607123454852647, 2.140297642852505, 2.3117428490210568, 1.6177193823404603, 2.037886405077575, 0.4105959084063113, 1.9655330065913432, 5.050243360472844, 0.6611550498937425, 1.681033313173775, 5.6325775627149595, 0.3747639256918957, 6.504707064887704, 2.0596179257328315, 2.8970793913871264, 2.0131828034234758, 2.3514535929930678, 1.0930274470478756, 2.615304762355622, 6.544842014899978, 1.885449018138646, 2.2436363341682632, 2.335853805356836, 6.125558097022672, 0.8673205866344911, 3.0606827669655687, 2.3280532639954767, 4.144421913849987, 2.5987520081762305, 1.0957376510825934, 2.0791209681016625, 5.970667131904105, 1.3680975842387904, 3.6132348387559867, 0.9127064150097743, 2.4096661594503113, 2.7918927271655694, 1.3264497728900235, 1.5535932543622863, 6.288688973069033, 0.13028046668629994, 1.4358962358053586, 2.28692479106769, 1.4689346479677052, 1.2887936219581453, 2.442820091615427, 1.1145447501110035, 2.2520810376183187, 2.8875446316896993, 2.1138242594880023, 4.818988898098853, 0.8223484662842171, 1.6274900921357385, 2.392791048127689, 1.0162721092305922, 0.9508533009881176, 1.9517799568598908, 1.440283999772267, 6.339813404194164, 2.08705366485867, 5.054647960046277, 6.65503012765532, 3.264270362577217, 1.7776079432765828, 2.091177897740889, 0.9750569214153606, 6.924875811160802, 5.563256690105178, 2.227414195878261, 1.0155692984725384, 2.8512534085906838, 1.2181896404090786, 2.0368519828401888, 2.9000689646972186, 0.5273338600924455, 1.1176215817529653, 2.7987077375102967, 1.9018554098563856, 2.8015656337126913, 1.3161200553141053, 5.884513743717486, 1.938602589495844, 1.6140734184045025, 1.7816534455387225, 5.682800805940676, 1.0254769621985675, 0.8138408935412372, 3.517446090560595, 2.4190080611688796, 2.7528628734464773, 2.6721807573590493, 2.1143727202175104, 1.5120958964298519, 1.5951332232763544, 1.2837188165638154, 1.3770192446004483, 1.0447109648127577, 2.299398617030113, 1.490313054361401, 2.165115701296353, 3.5056591391634213, 0.9869032374047596, 2.694215099059463, 3.7051613999932584, 2.747684843645646, 1.3685664032117677, 2.4963679616594976, 4.4372410346971245, 1.025312147592136, 3.745938734149292, 4.86381023478507, 3.3440671344935655, 1.9398706142420985, 1.1647596318554303, 4.514610725189937, 1.2768688264657406, 0.8383913167489278, 3.4990087167653643, 1.8586255674556924, 2.013785986643069, 4.601516815138243, 5.470797108283218, 0.84067175520532, 1.2255451848055217, 1.8864490451639544, 2.5225927138561275, 1.9937116140505409, 3.399991911755086, 1.5576215843394066, 5.843022676663167, 2.345249240485966, 3.3280013521631857, 2.4090224158359335, 6.567287720208396, 0.9729650558987214, 3.3294088063798974, 1.1905637320194156, 2.730224532891025, 1.6807260335938168, 1.5392277284404663, 0.9225193764902739, 1.8802284435674301, 4.735589931571355, 1.540520691195024, 1.215414744027733, 5.884428349466072, 0.3063249908185759, 1.723794071227766, 1.9141363587790687, 3.732543368803639, 1.9964663783795644, 6.240803634148408, 2.0348771461687787, 5.21414949919927, 1.3778773530325552, 1.2556826828462668, 2.38423551689006, 3.7716533509854826, 6.316323851735278, 1.817615470884863, 1.4892558544454333, 2.1409346089967345, 2.1235446781266467, 1.7290948499142582, 2.2645030359882505, 0.8829733857823827, 2.0989209132313658, 2.322242235426786, 1.594625034294896, 0.9535585980945291, 3.4761468611092936, 1.5486691060391178, 0.7502619542533124, 6.897527817993923, 3.800109735257654, 5.420950101227647, 2.924616897988522, 1.467435177443964, 1.5243742978678214, 1.637659610541823, 3.599580808927618, 1.8922877688131878, 3.944973637427758, 1.1934026981702373, 2.9699563969863236, 1.81473909970552, 2.3505797157297135, 2.0298674833594434, 1.6024843212961553, 0.9291571449437356, 5.029795820905656, 1.529042183852362, 5.634380977534267, 1.8642118978270694, 2.201916665089758, 2.4178333275889767, 4.2214901397492355, 1.7088537093619238, 6.266923487645273, 1.970822417164976, 1.1434775905106314, 2.486569725545617, 4.5971558598768425, 1.3820643979207332, 2.266750537663992, 3.499998571428279, 1.4445642941731613, 1.6747092284931147, 1.3329519871323192, 1.0817624508181092, 0.8041430221049004, 5.423102986298528, 0.7341409946325013, 1.558103013282498, 1.1875365257540482, 5.030561300690014, 1.6615670314495277, 2.3786407883495153, 1.189156844154716, 1.5614394000408705, 2.5704281744487636, 1.4137358310518964, 2.526710707619691, 4.57905547029079, 3.945701458549545, 0.8979710463038327, 5.930153539327627, 4.794499556783793, 1.0052964736832612, 1.3222458167829458, 1.7457242050220885, 1.562647433044319, 2.1415053116908203, 0.6139495093246671, 6.4982328367026065, 1.3925677721389362, 1.7843559062025722, 6.7019911966519325, 1.447759993921644, 1.9143312148110592, 6.664855662353088, 1.053965369450063, 6.411333402655022, 1.2946852899450108, 3.367947891520891, 1.956960398168546, 2.1864587807685747, 1.557846269694156, 1.7535774861693443, 2.4405019975406703, 1.2149049345524938, 2.733642441871287, 3.766944385042072, 5.352828971674697, 1.5128876362770656, 1.6763164975624376, 2.3853052215597046, 6.622600244616914, 2.5058900614352564, 0.7444575206148446, 2.26622218681223, 3.0785680112675764, 1.5638289548412923, 1.675413680259296, 2.2111530928454504, 1.1084678615097512, 2.009649720722494, 3.392657365546954, 2.511505723664592, 2.108407930169112, 2.4076613133910674, 2.5043721768139795, 3.7930047455810008, 3.2282121987254793, 1.646000000000002, 4.104974908571301, 0.9021252684633085, 1.3847158553291712, 1.7613702052663436, 0.9734274497876063, 1.5451177948622543, 0.5337387001145811, 4.655204399379261, 1.9118067370945218, 0.8925704453991284, 1.1684934745217896, 3.8853042351918856, 1.5040309172354158, 1.6811736971532705, 1.046550046581624, 1.6254922331404764, 5.6102667494514025, 3.868024172623537, 2.6309272129802452, 1.9250698688619097, 0.8225794794425645, 1.0653163849298464, 5.801404485122548, 5.0970499310875885, 3.855741303562776, 0.9479472559166972, 1.3413906962551954, 1.9682108626872274, 2.54877186111272, 0.7751554682771715, 1.3700142335027068, 5.0794899350230045, 0.6331982312040998, 1.8074780773221024, 1.251924917876468, 2.6448739100380565, 1.1430874857157658, 1.6249215365672265, 2.137072296390558, 1.7101017513586731, 2.3307451598147764, 1.8340387127866231, 1.8086608305594516, 2.237159806540426, 1.956112726813054, 0.9617764813094558, 3.2517289247414203, 3.083764906733326, 2.9205235831953162, 1.5610778327809263, 1.3668536863907554, 6.383787277157658, 6.295101428888972, 1.4423747779270133, 3.942349553248672, 0.23706117353965794, 1.3801728152662631, 1.7069812535584556, 3.990376172743617, 2.4279221569070124, 2.1729300494953807, 2.5767401886880243, 2.2535585193200554, 2.876475099840079, 5.7202337364831495, 2.365888627978927, 5.7843236423976165, 1.1412006834908608, 2.1167335684965174, 3.0673045169985995, 4.618853753909081, 2.8360622348601585, 1.1992335052023864, 5.800717628018106, 5.05416125583662, 0.7668128846074518, 2.0466428608821796, 6.908570474417986, 1.2929566891431414, 3.617012164756983, 3.467242420137364, 0.8936828296437163, 4.82408841129596, 2.2158542370832954, 1.5785623205942807, 4.04819441232755, 1.9054831408333164, 2.2387460776068395, 1.1578039557714426, 3.210732938131105, 1.8417461822954861, 3.6168009345276397, 4.777497357403769, 0.555032431484863, 1.5077735241076498, 3.4990087167653643, 2.0490012201070065, 6.191403475788019, 1.723196448464307, 1.5915693513007836, 1.5399720776689447, 2.1722032593659395, 2.220449954401133, 1.249773579493502, 1.3707961190490723, 2.692614714362232, 0.9531390244869844, 0.69357840220122, 0.9102549093523213, 6.0451124886142535, 0.6155014216068074, 1.8583920469050652, 1.105830457167826, 2.2781793169107645, 0.7048893530193229, 1.1680744839264319, 4.7659623372410325, 1.7358879572138293, 1.7935381233751344, 2.2165464127782215, 6.485663188911369, 1.3138576787460667, 5.707354991587608, 1.4623580272970094, 3.978159750437379, 1.5471735519973164, 3.090649769870405, 4.378389315718735, 6.557112855518045, 0.5724089447239669, 1.7645767197829614, 1.62609563064415, 3.646581001431341, 6.481419906162538, 1.6338708639301947, 1.8850339519488761, 0.8992674796744281, 1.5826231389689713, 1.5356044412543213, 5.408229469983684, 6.799022356192102, 1.6130895821373383, 2.453996943763378, 2.2184052830806205, 3.0831636349697673, 1.2578604851095394, 3.375354944298454, 2.520223204400754]
#
# folder_name='test_data'
# dist=[]
# for pro_label in pro_list:
#     for lig_label in lig_list:
#         pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
#         lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
#         pro = [pro_x_list, pro_y_list, pro_z_list]
#         lig = [lig_x_list, lig_y_list, lig_z_list]
#
#         min_d = float('inf')
#         is_contacted = False
#
#         for p in zip(pro[0], pro[1], pro[2]):
#             for l in zip(lig[0], lig[1], lig[2]):
#                 d_pl = np.sqrt(np.power(p[0]-l[0], 2)+np.power(p[1]-l[1], 2)+np.power(p[2]-l[2], 2))
#                 if d_pl < min_d:
#                     min_d = d_pl
#         dist.append(min_d)
#     print(dist)

loss =[0.6806, 0.4645, 0.3629, 0.3029, 0.2395, 0.1885, 0.154, 0.1365, 0.1086, 0.1298]

auc= [0.7590, 0.9213, 0.9543, 0.9641, 0.9752, 0.9834, 0.9872, 0.9922, 0.9934, 0.9964]
val_loss =[0.6076, 0.2308, 0.1962, 0.1806, 0.1994, 0.1718, 0.1390, 0.1897, 0.1454, 0.1916]
val_auc = [0.7970, 0.9633, 0.9750, 0.9814, 0.9726, 0.9814, 0.9900, 0.9847, 0.9878, 0.9836]

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range(1,11), auc)
plt.plot(range(1,11),val_auc)
plt.title('AUC vs epochs')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(range(1,11),loss)
plt.plot(range(1,11),val_loss)
plt.title('loss vs epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()