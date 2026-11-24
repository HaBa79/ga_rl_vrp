function [total_cost] = ObjectiveFunction(xy,Costt,V_cost,D_P,V_C)

mabda=length(D_P)+1;
% ساختن ماتریس نشان دادن ترتیب بازدید با استفاده از اعداد تصادفی

[~,j]=sort(xy(1:length(D_P)));
[~,q]=sort(xy(length(D_P)+1:end));

V_Order=zeros(length(V_C),length(D_P));

for p=1:length(V_C)
    if sum(j) ~= 0
        for w=1:length(D_P)
            if ( j(w) ~= 0 )
                if  ( V_C(q(p)) >= D_P(j(w)) )  
                    V_C(q(p)) = V_C(q(p))- D_P(j(w));
                    V_Order(q(p),w)=j(w);
                    j(w) = 0;
                end
            end
        end
    end
end

total_cost=0;
for r = 1:length(V_C)
    kC = nonzeros(V_Order(r,:)) ;
    if sum(kC) ~= 0
        k = [mabda , kC' , mabda] ;
        total_cost=total_cost+V_cost(r);
        for c =1:(length(k)-1)
            total_cost=total_cost+Costt(k(c),k(c+1),r);
        end
    end
end

end