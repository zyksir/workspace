#### Chapter 1 计算机网络

##### 什么是因特网

- A Nuts-and-Bolts Description：因特网是是一个连接这全世界计算设备的网络
  - host：手机、电脑等终端设备--》进一步可以分为 clients和servers
  - packets：传送的信息包
  - packet switch：路由器(used in network core)、link-layer switches(used in access networks)
  - Internet Service Providers (ISPs)：e.g 手机运营商、大学、酒店
  - 类似工厂之间运输运货，Internet 就是马路
- A Services Description：一个向应用提供服务的基础设施
  - 类似 Alice 给 Bob 写信，Internet 提供服务
- Protocal：描述了信息在两个交流实体之间交换的形式和顺序，也包括了信息传送/接受时的行为

- 家里一般都是依靠数字用户电线(DSL)和电缆联上网络的
  - DSL：当地电信公司一般把 DSL 和电话线一起部署，换句话说，其实DSLAM(对应的解调器)用的也是电话线传过来的数据，根据频率来区分传统电话信号和网络数据。-》CO：central office，DSL的另一端
  - 电缆：利用了电视公司已有的电缆。家庭利用同轴电缆连接到一个纤维节点(一般一个节点连接500-5000家)上，然后纤维节点通过纤维电缆连接到终端。电缆的调制解调器就是讲电流信号转化为数字形式。电缆网络的一个重要特性是其共享广播性-》最好大家是错峰使用；且我们需要一个协议来协调传送、避免碰撞
  - AON：之后讨论。和 PON 的共同点在于，CO 直接连一个纤维到家里，这样可以提供更高的速率。
  - PON：每家有一个optical network terminator (ONT), 多家的 ONT都连到一个 Optical splitter 上，然后Optical splitter和 CO 的optical line t­ erminator (OLT)相连。用户在家通过路由器和 ONT相连。PON 中，OLT 发出去的信息会被 splitter 复制。
- 



