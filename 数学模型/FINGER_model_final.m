%MCP设置1负责偏航，由于此处关节具有双自由度，我们设置L11的长度为0，将两个关节等效为一个。
theta_mcp_y = 0;
d11 = 0;
a11 = 0;
alpha11 = pi/2;

%MCP设置2负责俯仰
theta1 = 0;
d1 = 0;
a1 = 35;
alpha1 = 0;

%PIP
theta2 = 0;
d2 = 0;
a2 = 22;
alpha2 = 0;

%DIP
theta3 = 0;
d3 = 0;
a3 = 22;
alpha3 = 0;

%连杆与约束
%绘制模型
L11 = Link([theta_mcp_y,d11,a11,alpha11],'standard');
L1 = Link([theta1, d1, a1, alpha1],'standard'); 
L2 = Link([theta2, d2, a2, alpha2],'standard'); 
L3 = Link([theta3, d3, a3, alpha3],'standard');  
finger = SerialLink([L11,L1,L2,L3],'name','finger');
workspace = [-100 100 -100 100 -100 100];
finger.plot([theta_mcp_y,theta1,theta2,theta3],'workspace',workspace);
finger.teach();

% 蒙特卡洛法参数
num_samples = 30000; % 采样次数
workspace_points = zeros(num_samples, 3); % 存储末端执行器位置 (x, y, z)

% 关节角度限制
L11.qlim = [-pi/6, pi/6];
L1.qlim = [0 pi/2];   
L2.qlim = [0 11*pi/18];  
L3.qlim = [0, 0.37*pi];   
q_limits = [L11.qlim; L1.qlim; L2.qlim; L3.qlim];

% 蒙特卡洛随机采样
for i = 1:num_samples
    % 在关节限制范围内随机生成角度
    q = q_limits(:,1) + (q_limits(:,2) - q_limits(:,1)) .* rand(4,1);
    % 计算正运动学
    T = finger.fkine(q'); % 正运动学求解，得到末端执行器位姿
    workspace_points(i, :) = T.t'; % 提取平移部分 (x, y, z)
end

% 绘制工作空间
figure;
plot3(workspace_points(:,1), workspace_points(:,2), workspace_points(:,3), 'b.', 'MarkerSize', 1);
grid on;
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('食指工作空间（蒙特卡洛法，30,000 次采样）');
axis equal;


