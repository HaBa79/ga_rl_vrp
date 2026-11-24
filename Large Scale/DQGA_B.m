%% DQGA-B — Deep Q-Network Adaptive GA, Basic (Discrete State)
%  شبکه عصبی DQN به‌جای Q-table، اما فضای حالت مشابه Q-table‌ها (16 حالت one-hot)
%  نقش: مقایسه عادلانه Q-table vs DQN با همان تعریف حالت و اکشن
%  سوال: آیا DQN با همان اطلاعات ورودی بهتر از Q-table یاد می‌گیرد؟
%  نیاز: Deep Learning Toolbox (MATLAB)
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

%% ===== پارامترهای DQN =====
nStatesDQN = 16;    % one-hot encoding
nActions   = 12;

dqn_lr          = 1e-3;
dqn_gamma       = 0.9;
epsilon         = 0.3;   % کمتر از Q-table چون DQN سریع‌تر یاد می‌گیرد
replayBufSize   = 1000;
miniBatchSize   = 32;
minBufToTrain   = 64;
targetUpdateEvery = 30;
step            = 0.05;

action_map = [1 -1 -1; 1 -1 +1; 1 +1 -1; 1 +1 +1;
              2 -1 -1; 2 -1 +1; 2 +1 -1; 2 +1 +1;
              3 -1 -1; 3 -1 +1; 3 +1 -1; 3 +1 +1];

states = [1 1; 1 2; 1 3; 1 4;
          2 1; 2 2; 2 3; 2 4;
          3 1; 3 2; 3 3; 3 4;
          4 1; 4 2; 4 3; 4 4];

%% ===== ساخت شبکه DQN (ورودی: 16 one-hot) =====
layers = [
    featureInputLayer(nStatesDQN, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(nActions, 'Name', 'qValues')];

lgraph = layerGraph(layers);

%% ===== آرایه‌های خروجی =====
nRuns        = 10;
Conv_it      = zeros(1, nRuns);
Best_found   = zeros(1, nRuns);
Elapsed_time = zeros(1, nRuns);

empty_individual.Position = [];
empty_individual.Cost     = [];

for run = 1:nRuns
    fprintf('DQGA-B — Run %d/%d\n', run, nRuns);
    tic;

    %% بازنشانی شبکه‌ها و بافر برای هر اجرا
    qNet     = dlnetwork(lgraph);
    targetNet = dlnetwork(lgraph);
    targetNet.Learnables.Value = qNet.Learnables.Value;

    replay.s    = zeros(replayBufSize, nStatesDQN);
    replay.a    = zeros(replayBufSize, 1);
    replay.r    = zeros(replayBufSize, 1);
    replay.s2   = zeros(replayBufSize, nStatesDQN);
    replay.done = zeros(replayBufSize, 1);
    bufCount    = 0;
    trainSteps  = 0;

    mu_cur    = 0.1;
    gamma_cur = 0.4;
    UseRoulette   = false;
    UseTournament = false;
    UseRandom     = true;

    lastStateOH = [];
    lastAction  = [];

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

            %% --- تعریف حالت (one-hot 16 بُعدی) ---
            convergence = (std(Costs(~isnan(Costs))) / mean(Costs(~isnan(Costs)))) * 100;
            s1 = 1 + (convergence >= 0.9) + (convergence >= 1.8) + (convergence >= 2.7);
            s2 = min(4, floor(it / (MaxIt/4)) + 1);
            stateIdx = find(ismember(states, [s1 s2], 'rows'));
            curStateOH = zeros(1, nStatesDQN);
            curStateOH(stateIdx) = 1;

            %% --- محاسبه پاداش و ذخیره در بافر ---
            if it > 5
                reward = BestCost(it - 6) - BestCost(it - 1);
            else
                reward = 0;
            end

            if ~isempty(lastStateOH) && ~isempty(lastAction)
                bufCount = bufCount + 1;
                idx_buf  = mod(bufCount-1, replayBufSize) + 1;
                replay.s(idx_buf,:)  = lastStateOH;
                replay.a(idx_buf)    = lastAction;
                replay.r(idx_buf)    = reward;
                replay.s2(idx_buf,:) = curStateOH;
                replay.done(idx_buf) = 0;
            end

            %% --- آموزش DQN از replay buffer ---
            if bufCount >= minBufToTrain
                nb       = min(miniBatchSize, min(bufCount, replayBufSize));
                batchIdx = randperm(min(bufCount, replayBufSize), nb);

                batch_s    = replay.s(batchIdx,:);
                batch_a    = replay.a(batchIdx);
                batch_r    = replay.r(batchIdx);
                batch_s2   = replay.s2(batchIdx,:);
                batch_done = replay.done(batchIdx);

                % محاسبه هدف TD با target network
                dlXnext  = dlarray(batch_s2', 'CB');
                dlQnext  = predict(targetNet, dlXnext);
                qnext    = extractdata(dlQnext)';
                maxQnext = max(qnext, [], 2);
                targetVals = batch_r + dqn_gamma * (1 - batch_done) .* maxQnext;

                % ساخت targetAll
                dlXcur = dlarray(batch_s', 'CB');
                dlQcur = predict(qNet, dlXcur);
                qcur   = extractdata(dlQcur)';
                targetAll = qcur;
                for ii = 1:nb
                    targetAll(ii, batch_a(ii)) = targetVals(ii);
                end

                % آپدیت گرادیان
                [grads, ~] = dlfeval(@dqnGradients, qNet, batch_s, targetAll);
                qNet = dlupdate(@(p,g) p - dqn_lr .* g, qNet, grads);
                trainSteps = trainSteps + 1;

                if mod(trainSteps, targetUpdateEvery) == 0
                    targetNet.Learnables.Value = qNet.Learnables.Value;
                end
            end

            %% --- انتخاب اکشن (ε-greedy) ---
            if rand < epsilon
                action = randi([1, nActions]);
            else
                qVals_dl = predict(qNet, dlarray(curStateOH', 'CB'));
                qVals    = extractdata(qVals_dl);
                [~, action] = max(qVals);
            end

            %% --- اعمال اکشن ---
            sel_type  = action_map(action, 1);
            mu_dir    = action_map(action, 2);
            gamma_dir = action_map(action, 3);

            UseRoulette   = (sel_type == 1);
            UseTournament = (sel_type == 2);
            UseRandom     = (sel_type == 3);

            mu_cur    = max(0.01, min(1, mu_cur    + mu_dir    * step));
            gamma_cur = max(0.01, min(1, gamma_cur + gamma_dir * step));

            lastStateOH = curStateOH;
            lastAction  = action;
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

save('results_DQGA_B.mat', 'Best_found', 'Conv_it', 'Elapsed_time');
fprintf('\n===== DQGA-B Results =====\n');
fprintf('Best  : %.6f ± %.6f\n', mean(Best_found),   std(Best_found));
fprintf('Conv  : %.1f  ± %.1f\n', mean(Conv_it),      std(Conv_it));
fprintf('Time  : %.2fs ± %.2fs\n', mean(Elapsed_time), std(Elapsed_time));

figure; semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('DQGA-B'); grid on;

%% ===== تابع محاسبه گرادیان DQN =====
function [gradients, loss] = dqnGradients(qNet, states, targetAll)
    dlX      = dlarray(states', 'CB');
    dlTarget = dlarray(targetAll', 'CB');
    dlQ      = forward(qNet, dlX);
    diff     = dlQ - dlTarget;
    loss     = mean(diff.^2, 'all');
    gradients = dlgradient(loss, qNet.Learnables);
end
