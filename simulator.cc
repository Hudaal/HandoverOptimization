// Telenor Research
// Handover visualization experiment
// Vegard Edvardsen, 2021-2022

#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/wifi-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/applications-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("HandoverExperiment");

class HandoverExperiment
{
    public:
        HandoverExperiment(bool use_udp, int simulation_duration, int NeighbourCellOffset, int ServingCellThreshold, int enbCount, int ueCount, int max_speed, int min_speed, int x_pos, int y_pos, int rho);
        void enable_traces();
        void run();

    protected:
        bool use_udp;
        int simulation_duration;
        int enb_count;
        int ue_count;

        void create_lte_helpers(int NeighbourCellOffset, int ServingCellThreshold);
        Ptr<LteHelper> lte_helper;
        Ptr<EpcHelper> epc_helper;

        void create_lte_network();
        NodeContainer enb_nodes;

        void create_ue_nodes(int max_speed, int min_speed, int x_pos, int y_pos, int rho);
        NodeContainer ue_nodes;
        Ipv4InterfaceContainer ue_ifaces;

        void create_remote_server();
        NodeContainer server_nodes;
        Ipv4InterfaceContainer server_ifaces;

        void create_ue_applications();

        void dump_initial_state();
        void setup_callbacks();

        void periodically_dump_ue_state(
            Ptr<Node> node, PacketSizeMinMaxAvgTotalCalculator* packet_calc);
        static void callback_ipv4_packet_received(
            PacketSizeMinMaxAvgTotalCalculator* packet_calc,
            Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t iface);
        static void callback_ue_spotted_at_enb(
            std::string context, const uint64_t imsi,
            const uint16_t cell_id, const uint16_t rnti);
        static void callback_measurement_report_received(
            const uint64_t imsi, const uint16_t cell_id,
            const uint16_t rnti, const LteRrcSap::MeasurementReport report);

        inline int ue_node_id_to_imsi(int node_id)
        {
            // UEs are initialized after PGW, SGW, MME and all of the eNodeBs,
            // thus we subtract this amount and add back 1 to get the 1-indexed
            // IMSI. There is probably a proper way to do this
            return node_id - this->enb_count - 3 + 1;
        }
};

HandoverExperiment::HandoverExperiment(bool use_udp, int simulation_duration, int NeighbourCellOffset, int ServingCellThreshold, int enbCount, int ueCount, int max_speed, int min_speed, int x_pos, int y_pos, int rho)
    : use_udp(use_udp), simulation_duration(simulation_duration),
      enb_count(enbCount), ue_count(ueCount)
{
    this->create_lte_helpers(NeighbourCellOffset, ServingCellThreshold);
    this->create_lte_network();
    this->create_ue_nodes(max_speed, min_speed, x_pos, y_pos, rho);
    this->create_remote_server();
    this->create_ue_applications();
}

void HandoverExperiment::enable_traces()
{
    this->lte_helper->EnableTraces();
}

void HandoverExperiment::run()
{
    this->dump_initial_state();
    this->setup_callbacks();
    Simulator::Stop(Seconds(this->simulation_duration));
    Simulator::Run();
    Simulator::Destroy();
}

void HandoverExperiment::create_lte_helpers(int NeighbourCellOffset, int ServingCellThreshold)
{
    // Create LTE and EPC helpers. Network to be set up as a bunch of LTE base
    // stations (eNodeB), attached to an EPC (network core) implementation and
    // UEs (mobile handsets)
    this->epc_helper = CreateObject<PointToPointEpcHelper>();
    this->lte_helper = CreateObject<LteHelper>();
    this->lte_helper->SetEpcHelper(this->epc_helper);

    // Set up a directional antenna, to allow 3-sector base stations
    this->lte_helper->SetEnbAntennaModelType("ns3::ParabolicAntennaModel");
    this->lte_helper->SetEnbAntennaModelAttribute("Beamwidth", DoubleValue(70.0));

    // Activate handovers using a default RSRQ-based algorithm
    this->lte_helper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    this->lte_helper->SetHandoverAlgorithmAttribute ("NeighbourCellOffset", UintegerValue (NeighbourCellOffset));
    this->lte_helper->SetHandoverAlgorithmAttribute ("ServingCellThreshold", UintegerValue (ServingCellThreshold));

    // Specify that the RLC layer of the LTE stack should use Acknowledged Mode
    // (AM) as the default mode for all data bearers. This as opposed to the
    // ns-3 default which is Unacknowledged Mode (UM), see lte-enb-rrc.cc:1699
    // and lte-helper.cc:618. This is important because TCP traffic between a
    // UE and a remote host is very sensitive to packet loss. Packets lost
    // between the UE and eNodeB will be treated by TCP as a signal that the
    // network is congested -- but it might simply be that the radio conditions
    // are bad! RLC AM mode ensures reliable delivery across the radio link,
    // relieving TCP of that responsibility and not triggering any congestion
    // control algorithms in TCP. This greatly improves TCP performance
    Config::SetDefault("ns3::LteEnbRrc::EpsBearerToRlcMapping",
        EnumValue(LteEnbRrc::RLC_AM_ALWAYS));

    // Bump the maximum possible number of UEs connected per eNodeB
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(80));
}

void HandoverExperiment::create_lte_network()
{
    // Create topology helper that will place eNodeBs in a hexagonal pattern
    Ptr<LteHexGridEnbTopologyHelper> topo_helper = CreateObject<LteHexGridEnbTopologyHelper>();
    topo_helper->SetLteHelper(this->lte_helper);

    // Create eNodeB nodes using the topology helper, with default mobility
    this->enb_nodes.Create(this->enb_count);
    MobilityHelper mobility_helper;
    mobility_helper.Install(this->enb_nodes);
    topo_helper->SetPositionAndInstallEnbDevice(this->enb_nodes);

    // Add an X2 interface between all pairs of eNodeBs, to enable handovers
    this->lte_helper->AddX2Interface(this->enb_nodes);
}

void HandoverExperiment::create_ue_nodes(int max_speed, int min_speed, int x_pos, int y_pos, int rho)
{
    // Create UE nodes and add a default IP stack. The EPC helper will later
    // assign addresses to UEs in the 7.0.0.0/8 subnet by default
    this->ue_nodes.Create(this->ue_count);
    InternetStackHelper ip_stack_helper;
    ip_stack_helper.Install(this->ue_nodes);

    // Set up UE mobility (random starting points and random velocity vectors)
    MobilityHelper mobility_helper;
    mobility_helper.SetPositionAllocator("ns3::UniformDiscPositionAllocator",
        "X", DoubleValue(x_pos), "Y", DoubleValue(y_pos), "rho", DoubleValue(rho));
    mobility_helper.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(-600, 600, -400, 800)),
        "Speed", StringValue("ns3::UniformRandomVariable[Min="+to_string(min_speed)+"|Max="+to_string(max_speed)+"]"));
    mobility_helper.Install(this->ue_nodes);

    // Create UE net devices and assign IP addresses in EPC
    NetDeviceContainer ue_devices = this->lte_helper->InstallUeDevice(this->ue_nodes);
    this->ue_ifaces = this->epc_helper->AssignUeIpv4Address(ue_devices);

    // Attach the UEs to the LTE network
    this->lte_helper->Attach(ue_devices);

    // Set default IP route for all UEs
    Ipv4StaticRoutingHelper routing_helper;
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        routing_helper.GetStaticRouting(this->ue_nodes.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(this->epc_helper->GetUeDefaultGatewayAddress(), 1);
    }
}

void HandoverExperiment::create_remote_server()
{
    // Create the server that will send downlink traffic to UEs and respond to pings
    this->server_nodes.Create(1);
    InternetStackHelper ip_stack_helper;
    ip_stack_helper.Install(this->server_nodes);

    // Connect the server to the PDN gateway (PGW) in the EPC
    PointToPointHelper p2p_helper;
    p2p_helper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2p_helper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer server_devices = p2p_helper.Install(
        this->server_nodes.Get(0), this->epc_helper->GetPgwNode());

    // Set up IP interfaces on the link between PGW and the server
    Ipv4AddressHelper ipv4_helper("1.0.0.0", "255.0.0.0");
    this->server_ifaces = ipv4_helper.Assign(server_devices);

    // Add an IP route on the server toward the PGW interface, for the UE subnet (7.0.0.0/8)
    Ipv4StaticRoutingHelper routing_helper;
    Ptr<Ipv4StaticRouting> server_routing = routing_helper.GetStaticRouting(
        this->server_nodes.Get(0)->GetObject<Ipv4>());
    int server_iface_toward_pgw = this->server_ifaces.Get(0).second;
    server_routing->AddNetworkRouteTo("7.0.0.0", "255.0.0.0", server_iface_toward_pgw);
}

void HandoverExperiment::create_ue_applications()
{
    //// Set up a ping application on UE 0 toward the server at 1.0.0.1
    //V4PingHelper ping_helper(this->server_ifaces.GetAddress(0));
    //ping_helper.SetAttribute("Interval", TimeValue(Time("10ms")));
    //ping_helper.SetAttribute("Verbose", BooleanValue(true));
    //ApplicationContainer ping_apps = ping_helper.Install(this->ue_nodes.Get(0));
    //ping_apps.Start(Seconds(1));

    // Set up CBR (constant bitrate) traffic generators from the server to each UE
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        const char *socket_factory_type = (this->use_udp ?
            "ns3::UdpSocketFactory" : "ns3::TcpSocketFactory");
        InetSocketAddress cbr_dest(this->ue_ifaces.GetAddress(i), 10000);
        OnOffHelper cbr_helper(socket_factory_type, cbr_dest);
        cbr_helper.SetConstantRate(DataRate("10Mbps"));
        ApplicationContainer cbr_apps = cbr_helper.Install(this->server_nodes.Get(0));
        cbr_apps.Start(Seconds(1));

        // Set up a TCP/UDP sink on the receiving side (UE)
        PacketSinkHelper packet_sink_helper(socket_factory_type, cbr_dest);
        ApplicationContainer sink_apps = packet_sink_helper.Install(this->ue_nodes.Get(i));
        sink_apps.Start(Seconds(0));
    }
}

void HandoverExperiment::dump_initial_state()
{
    // Upon start of simulation, dump position and orientation of each eNodeB
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        Ptr<Node> node = this->enb_nodes.Get(i);
        int cell_id = node->GetDevice(0)->GetObject<LteEnbNetDevice>()->GetCellId();
        Vector position = node->GetObject<MobilityModel>()->GetPosition();
        int direction = (i % 3) * 120;
        std::cout << Simulator::Now().GetMilliSeconds() << " ms: Cell state: "
            << "Cell " << cell_id
            << " at " << position.x << " " << position.y
            << " direction " << direction << std::endl;
    }
}

void HandoverExperiment::setup_callbacks()
{
    // Periodically report the state of each UE. This involves printing to
    // stdout the current position of the UE, as well as the number of bytes
    // received in the last period. First we create a packet calculator that
    // will count the number of bytes, then we pass this to a method that will
    // reschedule itself for periodic reporting every 100 ms
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        PacketSizeMinMaxAvgTotalCalculator *packet_calc = new PacketSizeMinMaxAvgTotalCalculator();
        this->ue_nodes.Get(i)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext("Rx",
            MakeBoundCallback(&HandoverExperiment::callback_ipv4_packet_received, packet_calc));
        this->periodically_dump_ue_state(this->ue_nodes.Get(i), packet_calc);
    }

    // Connect callbacks to trigger whenever a UE is connected to a new eNodeB,
    // either because of initial network attachment or because of handovers
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
        MakeCallback(&HandoverExperiment::callback_ue_spotted_at_enb));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
        MakeCallback(&HandoverExperiment::callback_ue_spotted_at_enb));

    // Connect callback for whenever an eNodeB receives "measurement reports".
    // These reports contain signal strength information of neighboring cells,
    // as seen by a UE. This is used by the eNodeB to determine handovers
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/RecvMeasurementReport",
        MakeCallback(&HandoverExperiment::callback_measurement_report_received));
}

void HandoverExperiment::periodically_dump_ue_state(
    Ptr<Node> node, PacketSizeMinMaxAvgTotalCalculator* packet_calc)
{
    // Dump relevant simulation state for a given UE to stdout. Currently we
    // are interested in 2D position and IPv4 bytes received since last time
    Vector position = node->GetObject<MobilityModel>()->GetPosition();
    std::cout << Simulator::Now().GetMilliSeconds() << " ms: UE state: "
        << "IMSI " << this->ue_node_id_to_imsi(node->GetId())
        << " at " << position.x << " " << position.y
        << " with " << packet_calc->getSum() << " received bytes" << std::endl;

    // Reset the packet counter and reschedule again in 100 ms
    packet_calc->Reset();
    Simulator::Schedule(MilliSeconds(100),
        &HandoverExperiment::periodically_dump_ue_state,
        this, node, packet_calc);
}

void HandoverExperiment::callback_ipv4_packet_received(
    PacketSizeMinMaxAvgTotalCalculator* packet_calc,
    Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t iface)
{
    // Callback for each packet received at the IPv4 layer. Pass the packet
    // directly on to the PacketSizeMinMaxAvgTotalCalculator, which is used by
    // the periodic UE state reporting method above
    packet_calc->PacketUpdate("", packet);
}

void HandoverExperiment::callback_ue_spotted_at_enb(
    std::string context, const uint64_t imsi,
    const uint16_t cell_id, const uint16_t rnti)
{
    // A given eNodeB (identified by cell ID) has become responsible for an UE
    // (identified by its IMSI), due to initial network attachment or handover
    std::cout << Simulator::Now().GetMilliSeconds() << " ms: UE seen at cell: "
        << "Cell " << (int)cell_id << " saw IMSI " << imsi
        << " (context: " << context << ")" << std::endl;
}

void HandoverExperiment::callback_measurement_report_received(
    const uint64_t imsi, const uint16_t cell_id,
    const uint16_t rnti, const LteRrcSap::MeasurementReport report)
{
    // An eNodeB has received a measurement report of neighboring cell signal
    // strengths from an attached UE. Dump interesting information to stdout
    std::cout << Simulator::Now().GetMilliSeconds() << " ms: Measurement report: "
        << "Cell " << (int)cell_id
        << " got measurements from IMSI " << imsi
        << " (ID " << (int)report.measResults.measId
        << ", cell:RSRP/RSRQ " << (int)cell_id
        << ":" << (int)report.measResults.rsrpResult
        << "/" << (int)report.measResults.rsrqResult;

    // There might be additional measurements to the one listed directly in the
    // data structure, hence we need to do some additional iteration
    for (auto iter = report.measResults.measResultListEutra.begin();
            iter != report.measResults.measResultListEutra.end(); iter++) {
        std::cout << " " << (int)iter->physCellId << ":"
            << (int)iter->rsrpResult << "/" << (int)iter->rsrqResult;
    }
    std::cout << ")" << std::endl;
}


int main(int argc, char **argv) {
    // Parse command line arguments
    bool render_map = false;
    bool enable_traces = false;
    bool use_udp = false;
    int simulation_duration = 60;
    int ServingCellThreshold = 30;
    int NeighbourCellOffset = 1;
    int ueCount = 4;
    int enbCount = 5;
    int max_speed = 50;
    int min_speed = 10;
    int x_pos = 0;
    int y_pos = 300;
    int rho = 200;

    CommandLine cmd;
    cmd.AddValue("map", "Render the Radio Environment Map to 'rem.dat' and quit", render_map);
    cmd.AddValue("traces", "Enable traces from the LTE module to generate *Stats.txt files", enable_traces);
    cmd.AddValue("udp", "Use UDP instead of TCP for the 10 Mbps downlink traffic to each UE", use_udp);
    cmd.AddValue("duration", "Duration of the simulation in seconds", simulation_duration);
    cmd.AddValue("NeighbourCellOffset", "ns3::A2A4RsrqHandoverAlgorithm::NeighbourCellOffset attribute", NeighbourCellOffset);
    cmd.AddValue("ServingCellThreshold", "ns3::A2A4RsrqHandoverAlgorithm::ServingCellThreshold attribute", ServingCellThreshold);
    cmd.AddValue("UE_Count", "The number of all users", ueCount);
    cmd.AddValue("ENB_Count", "The number of all base stations", enbCount);
    cmd.AddValue("max_speed", "The maximum speed of users movment", max_speed);
    cmd.AddValue("min_speed", "The minimum speed of users movment", min_speed);
    cmd.AddValue("x_pos", "The start x position of users", x_pos);
    cmd.AddValue("y_pos", "The start y position of users", y_pos);
    cmd.AddValue("rho", "The rho of users", rho);
    cmd.Parse(argc, argv);

    // Initialize experiment (constructing network nodes, applications and callbacks)
    HandoverExperiment experiment(use_udp, simulation_duration, NeighbourCellOffset, ServingCellThreshold, enbCount, ueCount, max_speed, min_speed, x_pos, y_pos, rho);

    // If requested on command line, render a Radio Environment Map
    if (render_map) {
        Ptr<RadioEnvironmentMapHelper> rem_helper = CreateObject<RadioEnvironmentMapHelper>();
        rem_helper->SetAttribute("ChannelPath", StringValue("/ChannelList/2"));
        rem_helper->SetAttribute("XMin", DoubleValue(-750));
        rem_helper->SetAttribute("XMax", DoubleValue(750));
        rem_helper->SetAttribute("YMin", DoubleValue(-550));
        rem_helper->SetAttribute("YMax", DoubleValue(950));
        rem_helper->SetAttribute("OutputFile", StringValue("rem.dat"));
        rem_helper->Install();

        // Start the simulator to trigger the rendering, then quit
        Simulator::Run();
        Simulator::Destroy();
        return 0;
    }

    // If requested, enable trace outputs from the LTE module. This will
    // generate a bunch of Dl*Stats.txt/Ul*Stats.txt files in the working
    // directory, containing textual representations of most of the internal
    // trace sources supported by the LTE stack (such as within the PDCP/RLC
    // layers, MAC scheduling, UL/DL interference etc.)
    if (enable_traces) {
        experiment.enable_traces();
    }

    // Run the experiment and quit
    experiment.run();
    return 0;
}
