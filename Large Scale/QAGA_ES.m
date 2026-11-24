%% QAGA-ES — Q-learning Adaptive GA, Extended State Space
%  فضای حالت 4 بُعدی: {تنوع جمعیت, مرحله بهینه‌سازی, سطح mu, سطح gamma}
%  فضای اکشن: 12 (مثل QAGA-F)
%  نقش: آیا اطلاعات بیشتر در تعریف حالت → تصمیم بهتر RL؟
%  مقایسه اصلی: QAGA-ES vs QAGA-F (هر دو فضای اکشن=12، اما حالت متفاوت)
% ---------------------------------------------------------------
clear; close all;

load('Cost_3d_data_100_40.mat',   'Cost');
load('V_cost_3d_data_100_40.mat', 'V_cost');
load('D_P_3d_data_100_40.mat',    'D_P');
load('V_C_3d_data_100_40.mat',    'V_C');

%% ===== پارامترهای GA =====
nVar     = 140;
VarSize  = [1 nVar];
VarMin   = 0;    VarMax = 1;
MaxIt    = 500;
nPop     = 30;
pc       = 0.7;
nc       = 2*round(pc*nPop/2);
pm       = 0.3;
nm       = round(pm*nPop);
beta     = 8;
TournamentSize = 3;

%% ===== پارامترهای Q-learning با فضای حالت گسترده =====
% فضای حالت: 4^4 = 256 حالت (4 بُعد × 4 سطح هر بُعد)
%   بُعد 1 (s1): تنوع جمعیت (CV%): [0,0.9), [0.9,1.8), [1.8,2.7), [2.7,∞)
%   بُعد 2 (s2): مرحله بهینه‌سازی: [1..125], [126..250], [251..375], [376..500]
%   بُعد 3 (s3): سطح mu فعلی: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
%   بُعد 4 (s4): سطح gamma فعلی: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
nStates  = 256;
nActions = 12;
alpha_rl = 0.1;
epsilon  = 0.5;
step     = 0.05;

action_map = [1 -1 -1; 1 -1 +1; 1 +1 -1; 1 +1 +1;
              2 -1 -1; 2 -1 +1; 2 +1 -1; 2 +1 +1;
              3 -1 -1; 3 -1 +1; 3 +1 -1; 3 +1 +1];

%% ===== آرایه‌های خروجی =====
nRuns        = 10;
Conv_it      = zeros(1, nRuns);
Best_found   = zeros(1, nRuns);
Elapsed_time = zeros(1, nRuns);

empty_individual.Position = [];
empty_individual.Cost     = [];

for run = 1:nRuns
    fprintf('QAGA-ES — Run %d/%d\n', run, nRuns);
    tic;

    Q         = zeros(nStates, nActions);   % Q-table بزرگ‌تر
    mu_cur    = 0.1;
    gamma_cur = 0.4;
    UseRoulette   = false;
    UseTournament = false;
    UseRandom     = true;
    state  = 1;
    action = 9;

    pop = repmat(empty_individual, nPop, 1);
    for i = 1:nPop
        pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
        pop(i).Cost     = ObjectiveFunction(pop(i).Position, Cost, V_cost, D_P, V_C);
    end
    Costs = [pop.Cost];
    [Costs, idx] = sort(Costs);
    pop = pop(idx);
    BestCost  = zeros(MaxIt, 1);
    WorstCost = pop(end).Cost;

    for it = 1:MaxIt

        if rem(it, 5) == 0

            if it > 5
                reward = BestCost(it - 6) - BestCost(it - 1);
                Q(state, action) = Q(state, action) + alpha_rl * (reward - Q(state, action));
            end

            % --- تعریف حالت 4 بُعدی ---
            convergence = (std(Costs(~isnan(Costs))) / mean(Costs(~isnan(Costs)))) * 100;
            s1 = 1 + (convergence >= 0.9) + (convergence >= 1.8) + (convergence >= 2.7);
            s2 = min(4, floor(it / (MaxIt/4)) + 1);
            s3 = min(4, floor(mu_cur    / 0.25) + 1);
            s4 = min(4, floor(gamma_cur / 0.25) + 1);

            % کدگذاری خطی: شاخص در [1, 256]
            state = (s1-1)*64 + (s2-1)*16 + (s3-1)*4 + (s4-1) + 1;

            if rand < epsilon
                action = randi([1, nActions]);
            else
                [~, action] = max(Q(state, :));
            end

            sel_type  = action_map(action, 1);
            mu_dir    = action_map(action, 2);
            gamma_dir = action_map(action, 3);

            UseRoulette   = (sel_type == 1);
            UseTournament = (sel_type == 2);
            UseRandom     = (sel_type == 3);

            mu_cur    = max(0.01, min(1, mu_cur    + mu_dir    * step));
            gamma_cur = max(0.01, min(1, gamma_cur + gamma_dir * step));
        end

        %% --- Crossover ---
        if UseRoulette
            P = exp(-beta * Costs / WorstCost);
            P = P / sum(P);
        end
        popc = repmat(empty_individual, nc/2, 2);
        for k = 1:nc/2
            if UseRoulette
                i1 = RouletteWheelSelection(P);
                i2 = RouletteWheelSelection(P);
            elseif UseTournament
                i1 = TournamentSelection(pop, TournamentSize);
                i2 = TournamentSelection(pop, TournamentSize);
            else
                i1 = randi([1 nPop]);
                i2 = randi([1 nPop]);
            end
            [popc(k,1).Position, popc(k,2).Position] = ...
                Crossover(pop(i1).Position, pop(i2).Position, gamma_cur, VarMin, VarMax);
            popc(k,1).Cost = ObjectiveFunction(popc(k,1).Position, Cost, V_cost, D_P, V_C);
            popc(k,2).Cost = ObjectiveFunction(popc(k,2).Position, Cost, V_cost, D_P, V_C);
        end
        popc = popc(:);

        %% --- Mutation ---
        popm = repmat(empty_individual, nm, 1);
        for k = 1:nm
            p = pop(randi([1 nPop]));
            popm(k).Position = Mutate(p.Position, mu_cur, VarMin, VarMax);
            popm(k).Cost     = ObjectiveFunction(popm(k).Position, Cost, V_cost, D_P, V_C);
        end

        pop   = [pop; popc; popm];
        Costs = [pop.Cost];
        [Costs, idx] = sort(Costs);
        pop       = pop(idx);
        WorstCost = max(WorstCost, pop(end).Cost);
        pop       = pop(1:nPop);
        Costs     = Costs(1:nPop);
        BestCost(it) = pop(1).Cost;
    end

    Elapsed_time(run) = toc;
    Best_found(run)   = BestCost(end);
    tmp               = find(BestCost == BestCost(end), 1, 'first');
    Conv_it(run)      = tmp;
end

save('results_QAGA_ES.mat', 'Best_found', 'Conv_it', 'Elapsed_time');
fprintf('\n===== QAGA-ES Results =====\n');
fprintf('Best  : %.6f ± %.6f\n', mean(Best_found),   std(Best_found));
fprintf('Conv  : %.1f  ± %.1f\n', mean(Conv_it),      std(Conv_it));
fprintf('Time  : %.2fs ± %.2fs\n', mean(Elapsed_time), std(Elapsed_time));

figure; semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('QAGA-ES'); grid on;
