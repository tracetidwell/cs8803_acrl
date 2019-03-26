import py4j.GatewayServer;

public class StateEntryPoint {

    private State state;

    public StateEntryPoint() {
        state = new State(24);
    }

    public State getState() {
        return state;
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new StateEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }

}
