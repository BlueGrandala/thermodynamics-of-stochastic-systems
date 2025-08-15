using LinearAlgebra
using Plots
using LaTeXStrings
using Graphs
using GraphRecipes

mutable struct graph 
    E::Vector{Float64}
    connection::Matrix{Float64}
    disspate::Matrix{Float64} #默认为0
    weight::Matrix{Float64}   #its wij.
    fixed::Matrix{Float64}    #fixed weights
    barrier::Matrix{Float64}  #barriers
end

function init(E::Vector{Float64})
    n = length(E)
    connection = Matrix{Float64}(I, n, n) # 创建单位矩阵
    disspate = zeros(Float64, n, n)
    weight = zeros(Float64, n, n)
    fixed = zeros(Float64, n, n)
    barrier = zeros(Float64, n, n)
    return graph(E, connection, disspate, weight, fixed, barrier)
    
end

function L(Graph::graph, dis_drive, beta)
        E = Graph.E
        disspate = Graph.disspate
        connection = Graph.connection
        n = length(E)
        
        # 1. 构建 rate 矩阵 (这是唯一被替换的部分)
        #    使用与我们Python代码一致的对称速率
        wij = (E' .- E) .+ dis_drive .* disspate
        rate = 1 ./(1 .+ exp.(beta .* wij ) )
        rate = rate .* connection
        for i in 1:n rate[i, i] = 0 end # 确保对角线为0

        # 2. 构建生成元 L (与您的原始逻辑相同)
        row_sums = sum(rate, dims=2)
        d = Diagonal(vec(row_sums))
        L = rate - d
    return L
end


function first_passage(L, x::Int)
    L_H = copy(L)
    L_H[x,:].= 0.0
    L_H[ x, x]=1.0
    f = ones(size(L)[1])
    f[x] = 0.0
    return(-L_H\f)
end

function get_V(Graph::graph, dis_drive, current_beta)
        E = Graph.E
        disspate = Graph.disspate
        connection = Graph.connection
        n = length(E)
        
        # 1. 构建 rate 矩阵 (这是唯一被替换的部分)
        #    使用与我们Python代码一致的对称速率
        wij = (E' .- E) .+ dis_drive .* disspate
        rate = 1 ./(1 .+ exp.(current_beta .* wij ) )
        rate = rate .* connection
        for i in 1:n rate[i, i] = 0 end # 确保对角线为0

        # 2. 构建生成元 L (与您的原始逻辑相同)
        row_sums = sum(rate, dims=2)
        d = Diagonal(vec(row_sums))
        L = rate - d

        # 3. 求解稳态 rho (与您的原始逻辑相同)
        cof = [L'; ones(1, n)]
        b = vcat(zeros(n), [1.0])
        ρ = cof \ b

        # 4. 计算源项 f (Q) (与您的原始逻辑相同)

        function power_simplified(x, L, wij)
             return -dot(L[x, :], wij[x, :])
        end
        
        powers = [power_simplified(x, L, wij) for x in 1:n]
        #powers = [power(x) for x in 1:n]
        ave_power = sum(powers[i] * ρ[i] for i in 1:n)
        f = [powers[i] - ave_power for i in 1:n]

        # 5. 求解泊松方程 LV = -f (与您的原始逻辑相同)
        #    注意：这里使用了-f，遵循物理定义，使结果更具可比性
        L_aug = [L; ρ']
        source = vcat(-f, [0.0])
        v = L_aug \ source
        
        return v, ρ
    end




function heatCapacity_on_graph(Graph::graph, dis_drive, T)
    """
    使用您原始代码的核心算法，计算给定图体系的非平衡热容。
    """
    β = 1.0 / T
    δ = 1e-5 # 用于数值微分的步长
    # --- 主计算流程 (与您的原始逻辑相同) ---
    v_beta, ρ_beta = get_V(Graph,dis_drive,β)
    v_beta_plus_delta, _ = get_V(Graph,dis_drive,β + δ)
    
    dV_dβ = (v_beta_plus_delta - v_beta) ./ δ
    
    C = β^2 * dot(dV_dβ, ρ_beta)
    return C
end


function get_V_from_L(Lij::Matrix,current_beta)
    n = size(Lij)[1]
    L = Lij
    cof = [L'; ones(1, n)]
    b = vcat(zeros(n), [1.0])
    ρ = cof \ b                 #returns distribution

        function power(x)
            s = 0.0
            for y in 1:n
                if x != y && L[x, y] > 1e-15 && L[y, x] > 1e-15
                    s += log(L[x, y] / L[y, x]) * L[x, y]
                end
            end
            return s / current_beta
        end

    #powers = [power_simplified(x, L, wij) for x in 1:n]
    powers = [power(x) for x in 1:n]
    ave_power = sum(powers[i] * ρ[i] for i in 1:n)
    f = [powers[i] - ave_power for i in 1:n]
    
    L_aug = [L; ρ']
    source = vcat(-f, [0.0])
    v = L_aug \ source
        
    return v, ρ
end


function heatCapacity_by_L(L1::Function, T)  
    β = 1.0 / T
    δ = 1e-5 # 用于数值微分的步长

    v_beta, ρ_beta = get_V_from_L(L1(β),β)
    v_beta_plus_delta, _ = get_V_from_L(L1(β + δ),β + δ)
    
    dV_dβ = (v_beta_plus_delta - v_beta) ./ δ
    C = β^2 * dot(dV_dβ, ρ_beta)
    return C
end


function get_Leff(L::Matrix{Float64},slow_indices, fast_indices)
    if size(L)[1] != size(slow_indices)[1]+size(fast_indices)[1]
        error("Dimension error.")
    end

    # 3. 从 L 中提取四个子矩阵块
    L_SS = L[slow_indices, slow_indices]
    L_SF = L[slow_indices, fast_indices]
    L_FS = L[fast_indices, slow_indices]
    L_FF = L[fast_indices, fast_indices]

    # 4. 检查快空间的矩阵 L_FF 是否可逆
    if abs(det(L_FF)) < 1e-15
        error("快空间矩阵 L_FF 是奇异的，无法进行绝热消除。")
    end
    
    # 5. 应用绝热消除公式计算微扰项
    # 注意：根据我们对经典主方程的严谨推导，这里的符号是负号
    L_pert = -L_SF * inv(L_FF) * L_FS

    # 6. 计算最终的有效生成元
    L_eff = L_SS + L_pert
    
    return L_eff, L_pert
end


function genfig()
    fig = plot(
    tickfont = Plots.font(11),
    guidefont = Plots.font(13),
    legendfont = Plots.font(11),
    title = "plot",
    xlabel = L"T",
    ylabel = L"X(T)"
    )
    return fig
end


function first_passage(L, x::Vector{Int64})  #X：Escape to x
    L_H = copy(L)
    f = ones(size(L)[1])
    for i in x
        L_H[i,:].= 0.0
        L_H[ i, i]=1.0
        f[i] = 0.0
    end
    return(-L_H\f)
end


function check_drive(g::graph,X,Y)
    Lij = L(g,0.0,1)[X,Y]
    Lij1 = L(g,0.1,1)[X,Y]
    if Lij1>Lij
        println("driving along $X to $Y")
    else 
        println("driving against $X to $Y")
    end
    return 
end

function vis(g::graph)
    n = length(g.E)
    C = g.connection
    C[diagind(C)] .= 0
    gr = SimpleGraph(C)

    p = plot(gr,
    # --- 核心：在这里标注数值 ---
    names = g.E, # 使用 names 关键字来设置节点标签

    fontsize = 12,
    nodeshape = :circle,
    nodesize = 0.1,
    linecolor = :gray,
    method = :spring, # 布局算法
    title = "Visualization"
    )
    return  p
end