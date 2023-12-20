EXECS=hello_world send_recv ping_pong
MPICC?=mpicc

all: ${EXECS}

hello_world: hello_world.c
	${MPICC} -o hello_world hello_world.c

send_recv: send_recv.c
	${MPICC} -o send_recv send_recv.c

ping_pong: ping_pong.c
	${MPICC} -o ping_pong ping_pong.c

clean:
	rm -f ${EXECS}
