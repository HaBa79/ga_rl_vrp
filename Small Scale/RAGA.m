%% RAGA — Random-Adaptive Genetic Algorithm
%  پارامترها به‌صورت تصادفی تغییر می‌کنند، بدون هیچ یادگیری‌ای
%  نقش: جداسازی اثر «تطبیق» از اثر «یادگیری» در الگوریتم‌های RL-GA
%  سوال: آیا صرف تغییر پارامتر (بدون هدایت RL) هم مفید است؟
% ---------------------------------------------------------------
clc; clear; close all;

load('Cost_3d_data_10_4.mat',   'Cost');
load('V_cost_3d_data_10_4.mat', 'V_cost');
load('D_P_3d_data_10_4.mat',    'D_P');
load('V_C_3d_data_10_4.mat',    'V_C');

%% ===== پارامترهای پایه GA =====
nVar     = 14;
VarSize  = [1 nVar];
VarMin   = 0;    VarMax = 1;

MaxIt    = 500;
nPop     = 30;
pc       = 0.7;
nc       = 2*round(pc*nPop/2);
gamma    = 0.4;
pm       = 0.3;
nm       = round(pm*nPop);
mu       = 0.1;

beta           = 8;
TournamentSize = 3;

% فاصله تصمیم‌گیری (هر 5 تکرار مانند نسخه‌های Q-learning)
decisionInterval = 5;

%% ===== آرایه‌های خروجی =====
nRuns        = 10;
Conv_it      = zeros(1, nRuns);
Best_found   = zeros(1, nRuns);
Elapsed_time = zeros(1, nRuns);

empty_individual.Position = [];
empty_individual.Cost     = [];

for run = 1:nRuns
    fprintf('RAGA — Run %d/%d\n', run, nRuns);
    tic;

    % بازنشانی پارامترها به مقادیر اولیه
    mu_cur    = mu;
    gamma_cur = gamma;
    UseRoulette    = false;
    UseTournament  = false;
    UseRandom      = true;

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

        %% --- تصمیم تصادفی (بدون RL) هر decisionInterval تکرار ---
        if rem(it, decisionInterval) == 0
            % انتخاب تصادفی روش انتخاب
            sel = randi([1 3]);
            UseRoulette   = (sel == 1);
            UseTournament = (sel == 2);
            UseRandom     = (sel == 3);

            % تغییر تصادفی نرخ جهش (گام ثابت ±0.05)
            if rand < 0.5
                mu_cur = min(1, mu_cur + 0.05);
            else
                mu_cur = max(0, mu_cur - 0.05);
            end

            % تغییر تصادفی extra-range کراس‌اور
            if rand < 0.5
                gamma_cur = min(1, gamma_cur + 0.05);
            else
                gamma_cur = max(0, gamma_cur - 0.05);
            end
        end

        % --- Crossover ---
        nc_cur = 2*round(pc*nPop/2);
        if UseRoulette
            P = exp(-beta * Costs / WorstCost);
            P = P / sum(P);
        end
        popc = repmat(empty_individual, nc_cur/2, 2);
        for k = 1:nc_cur/2
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

        % --- Mutation ---
        nm_cur = round(pm*nPop);
        popm   = repmat(empty_individual, nm_cur, 1);
        for k = 1:nm_cur
            p = pop(randi([1 nPop]));
            popm(k).Position = Mutate(p.Position, mu_cur, VarMin, VarMax);
            popm(k).Cost     = ObjectiveFunction(popm(k).Position, Cost, V_cost, D_P, V_C);
        end

        % --- ادغام، مرتب‌سازی، برش ---
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

%% ===== ذخیره و نمایش =====
save('results_RAGA.mat', 'Best_found', 'Conv_it', 'Elapsed_time');
fprintf('\n===== RAGA Results =====\n');
fprintf('Best  : %.6f ± %.6f\n', mean(Best_found),   std(Best_found));
fprintf('Conv  : %.1f  ± %.1f\n', mean(Conv_it),      std(Conv_it));
fprintf('Time  : %.2fs ± %.2fs\n', mean(Elapsed_time), std(Elapsed_time));

figure; semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('RAGA'); grid on;
