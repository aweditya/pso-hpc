#include <stdio.h>
#include <unistd.h>

int main()
{
	char* argv[] = {"Command-line", "test.cir", NULL};
	int pid = fork();
	if ( pid == 0)
	{
		printf("Inside child process\n");
		printf("Process ID (pid) of child is: %d\n", (int)getpid());
		execvp("ngspice", argv);
	}
	else
	{
		printf("Inside parent process - pid of child is %d\n", pid);
		printf("Process ID (pid) of parent is: %d\n", (int)getpid());
	}
	return 0;
}
