pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh '''#!/bin/bash
                echo 'In C or Java, we can compile our program in this step'
                echo 'In Python, we can build our package here or skip this step'
                '''
            }
        }
        stage('Test') {
            steps {
                sh '''#!/bin/bash
                echo 'Test Step: We run testing tool like pytest here'

                # TODO fill out the path to conda here
                # sudo /PATH/TO/CONDA init
                # /home/santyg20/miniconda3/bin/conda init bash
                #source ~/.bashrc
                export PATH=/home/santyg20/miniconda3/bin:$PATH
                # TODO Complete the command to run pytest
                # sudo /PATH/TO/CONDA run -n <Envinronment Name> <Command you want to run>
                conda run -n mlip pytest

                echo 'pytest runned'
                #exit 1 #comment this line after implementing Jenkinsfile
                '''

            }
        }
        stage('Deploy') {
            steps {
                echo 'In this step, we deploy our porject'
                echo 'Depending on the context, we may publish the project artifact or upload pickle files'
            }
        }
    }
}

