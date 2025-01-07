// Manages communication with github for reporting important status
// events. Relies on Github integration plugin https://github.com/jenkinsci/github-plugin
class GithubStatus {

    // Repository to send status e.g: '/username/repo'
    String Repository

    // Commit hash to send status e.g: `909b76f97e`
    String Commit_id

    // Build url send status e.g: `https://jenkins/build/123`
    String BUILD_URL

    // Set pending status for this pull request
    def setPending(script, String context_name, String result_status_when_set_commit_failure = "UNSTABLE") {
        this.setBuildStatusStep(script, context_name, "In Progress", "PENDING", result_status_when_set_commit_failure)
    }

    // Set failed status for this pull request
    def setFailed(script, String context_name, String result_status_when_set_commit_failure = "UNSTABLE") {
        this.setBuildStatusStep(script, context_name, "Complete", "FAILURE", result_status_when_set_commit_failure)
    }

    // Set success status for this pull request
    def setSuccess(script, String context_name, String result_status_when_set_commit_failure = "UNSTABLE") {
        this.setBuildStatusStep(script, context_name, "Complete", "SUCCESS", result_status_when_set_commit_failure)
    }

    // Manually set PR status via GitHub integration plugin. Manual status updates are needed
    // Due to pipeline stage retries not propagating updates back to Github automatically
    def setBuildStatusStep(script, String context_name, String message, String state, String result_status_when_set_commit_failure) {
        def commitContextName = context_name
        def build_url_secret = this.BUILD_URL.replace("icl-jenkins.sc", "llvm-ci")

        script.retry(4) {
            script.step([
                    $class             : "GitHubCommitStatusSetter",
                    reposSource        : [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/${this.Repository}"],
                    contextSource      : [$class: "ManuallyEnteredCommitContextSource", context: commitContextName],
                    errorHandlers      : [],
                    commitShaSource    : [$class: "ManuallyEnteredShaSource", sha: this.Commit_id],
                    //statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: "${this.BUILD_URL}flowGraphTable/"],
                    statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: "${build_url_secret}"],
                    statusResultSource : [$class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]]]
            ])
        }


    }
}


def fill_task_name_description (String oneapi_package_date = "Default") {
    script {
        short_commit_sha = env.Commit_id.substring(0,10)
        currentBuild.displayName = "PR#${env.PR_number}-No.${env.BUILD_NUMBER}"
        currentBuild.description = "PR number: ${env.PR_number} / Commit id: ${short_commit_sha} / Oneapi package date: ${oneapi_package_date}"
    }
}

def githubStatus = new GithubStatus(
        Repository: params.Repository,
        Commit_id: params.Commit_id,
        BUILD_URL: env.RUN_DISPLAY_URL
)

def runExample(String test_name, String cmake_flags = "") {
    def result = sh(
        script: "cd ./src/examples/" + test_name + "/ && mkdir build && cd build/ && cmake " + cmake_flags + " .. && make && make run && exit \$?",
        returnStatus: true, label: test_name + "_test Step")
    sh(
        script: "rm -r ./src/examples/" + test_name + "/build")
    if (result != 0) {
        echo test_name + " check failed."
        return false
    }
    else {
        return true
    }
}

String pipeline_name = "Jenkins/RHEL_Check (dpcpp_gpu_cxx_17, dpcpp_fpga_emu_cxx_17)"
build_ok = true
fail_stage = ""
user_in_github_group = false
boolean code_changed

pipeline {

    agent { label "oneDPL_scheduler" }
    options {
        durabilityHint 'PERFORMANCE_OPTIMIZED'
        timeout(time: 5, unit: 'HOURS')
        timestamps()
    }

    environment {
        def NUMBER = sh(script: "expr ${env.BUILD_NUMBER}", returnStdout: true).trim()
        def TIMESTAMP = sh(script: "date +%s", returnStdout: true).trim()
        def DATESTAMP = sh(script: "date +\"%Y-%m-%d\"", returnStdout: true).trim()
        def TEST_TIMEOUT = 900
    }

    parameters {
        string(name: 'Commit_id', defaultValue: 'None', description: '',)
        string(name: 'PR_number', defaultValue: 'None', description: '',)
        string(name: 'Repository', defaultValue: 'uxlfoundation/oneDPL', description: '',)
        string(name: 'User', defaultValue: 'None', description: '',)
        string(name: 'OneAPI_Package_Date', defaultValue: 'Default', description: '',)
        string(name: 'Base_branch', defaultValue: 'main', description: '',)
    }

    triggers {
        GenericTrigger(
                genericVariables: [
                        [key: 'Commit_id', value: '$.pull_request.head.sha', defaultValue: 'None'],
                        [key: 'PR_number', value: '$.number', defaultValue: 'None'],
                        [key: 'Repository', value: '$.pull_request.base.repo.full_name', defaultValue: 'None'],
                        [key: 'User', value: '$.pull_request.user.login', defaultValue: 'None'],
                        [key: 'action', value: '$.action', defaultValue: 'None'],
                        [key: 'Base_branch', value: '$.pull_request.base.ref', defaultValue: 'main']
                ],

                causeString: 'Triggered on $PR_number',

                token: 'oneDPL-pre-ci',

                printContributedVariables: true,
                printPostContent: true,

                silentResponse: false,

                regexpFilterText: '$action',
                regexpFilterExpression: '(opened|reopened|synchronize)'
        )
    }

    stages {
        stage('Check_User_in_Org') {
            agent {
                label "oneDPL_scheduler"
            }
            steps {
                script {
                    try {
                        retry(2) {
                            fill_task_name_description()
                            def check_user_return = sh(script: "python3 /export/users/oneDPL_CI/check_user_in_group.py -u  ${env.User}", returnStatus: true, label: "Check User in Group")
                            echo "check_user_return value is $check_user_return"
                            if (check_user_return == 0) {
                                user_in_github_group = true
                                if (env.OneAPI_Package_Date == "Default") {
                                    sh(script: "bash /export/users/oneDPL_CI/get_good_compiler.sh ", label: "Get good compiler stamp")
                                    if (fileExists('./Oneapi_Package_Date.txt')) {
                                        env.OneAPI_Package_Date = readFile('./Oneapi_Package_Date.txt')
                                    }
                                }
                                echo "Oneapi package date is: " + env.OneAPI_Package_Date.toString()
                                fill_task_name_description(env.OneAPI_Package_Date)
                                githubStatus.setPending(this, pipeline_name)
                            }
                            else {
                                user_in_github_group = false
                                currentBuild.result = 'UNSTABLE'
                            }
                        }
                    }
                    catch (e) {
                        fail_stage = fail_stage + "    " + "Check_User_in_Org"
                        user_in_github_group = false
                        echo "Exception occurred when check User:${env.User} in group. Will skip build this time"
                        sh script: "exit -1", label: "Set Failure"
                    }
                }
            }
        }

        stage('RHEL_Check') {
            when {
                expression { user_in_github_group }
            }
            agent { label "oneDPL_RHEL" }
            stages {
                stage('Git-monorepo') {
                    steps {
                        script {
                            try {
                                retry(2) {
                                    deleteDir()
                                    if (fileExists('./src')) {
                                        sh script: 'rm -rf src', label: "Remove Src Folder"
                                    }
                                    sh script: 'cp -rf /export/users/oneDPL_CI/oneDPL-src/src ./', label: "Copy src Folder"
                                    sh script: "cd ./src; git config --local --add remote.origin.fetch +refs/pull/${env.PR_number}/head:refs/remotes/origin/pr/${env.PR_number}", label: "Set Git Config"
                                    sh script: "cd ./src; git pull origin; git checkout ${env.Commit_id}; git merge origin/${env.Base_branch}", label: "Checkout Commit"
                                }
                            }
                            catch (e) {
                                build_ok = false
                                fail_stage = fail_stage + "    " + "Git-monorepo"
                                sh script: "exit -1", label: "Set failure"
                            }
                        }
                    }
                }

                stage('Check_code_changes') {
                    steps {
                        script {
                            dir("./src") {
                                code_changed = sh(
                                    script: "! git diff --name-only main | grep -v ^documentation",
                                    returnStatus: true, label: "Code_changed")
                            }
                        }
                    }
                }

                stage('Setting_Env') {
                    when {
                        expression { code_changed }
                    }
                    steps {
                        script {
                            try {
                                sh script: """
                                    bash /export/users/oneDPL_CI/generate_env_file.sh ${env.OneAPI_Package_Date}
                                    if [ ! -f ./envs_tobe_loaded.txt ]; then
                                        echo "Environment file not generated."
                                        exit -1
                                    fi
                                    cd ${env.OneAPI_Package_Date}
                                    cp -rf ../src/include/oneapi ./build/linux_prod/dpcpp-ct/include
                                """, label: "Generate environment vars"
                            }
                            catch (e) {
                                build_ok = false
                                fail_stage = fail_stage + "    " + "Setting_Env"
                                sh script: "exit -1", label: "Set failure"
                            }
                        }
                    }
                }
                
                stage('Tests_dpcpp_gpu_cxx_17') {
                    when {
                        expression { code_changed }
                    }
                    steps {
                        timeout(time: 2, unit: 'HOURS') {
                            script {
                                try {
                                    dir("./src/build") {
                                        withEnv(readFile('../../envs_tobe_loaded.txt').split('\n') as List) {
                                            sh script: """
                                                rm -rf *
                                                export PATH=/usr/bin/:$PATH
                                                dpcpp --version
                                                cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_CXX_STANDARD=17 -DONEDPL_BACKEND=dpcpp -DONEDPL_DEVICE_TYPE=GPU -DCMAKE_BUILD_TYPE=release ..
                                                make VERBOSE=1 build-onedpl-tests -j`nproc` -k || true
                                                ctest --output-on-failure --timeout ${TEST_TIMEOUT}
                                            """, label: "all tests"
                                        }

                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    echo "Exception is" + e.toString()
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        sh script: """
                                            exit -1
                                        """
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Tests_dpcpp_fpga_emu_cxx_17') {
                    when {
                        expression { code_changed }
                    }
                    steps {
                        timeout(time: 2, unit: 'HOURS') {
                            script {
                                try {
                                    dir("./src/build") {
                                        withEnv(readFile('../../envs_tobe_loaded.txt').split('\n') as List) {
                                            def TESTS="merge.pass copy_if.pass remove_copy.pass adjacent_find.pass all_of.pass equal.pass for_each.pass minmax_element.pass set.pass partial_sort_copy.pass sort.pass transform_reduce.pass transform_scan.pass"
                                            sh script: """
                                                rm -rf *
                                                export PATH=/usr/bin/:$PATH
                                                dpcpp --version
                                                cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_CXX_STANDARD=17 -DONEDPL_BACKEND=dpcpp -DONEDPL_DEVICE_TYPE=FPGA_EMU -DCMAKE_BUILD_TYPE=release ..
                                                make VERBOSE=1 ${TESTS} -k || true
                                                ctest  -R \"^(`echo ${TESTS} | sed 's/ /|/g'`)\$\" --output-on-failure --timeout ${TEST_TIMEOUT}
                                            """, label: "run FPGA test suite"
                                        }

                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    echo "Exception is" + e.toString()
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        sh script: """
                                            exit -1
                                        """
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Check_Samples') {
                    when {
                        expression { code_changed }
                    }
                    steps {
                        timeout(time: 1, unit: 'HOURS'){
                            script {
                                try {
                                    withEnv(readFile('envs_tobe_loaded.txt').split('\n') as List) {
                                        String[] examples_dpcpp = ["gamma_correction","stable_sort_by_key","convex_hull","dot_product","histogram","random"]
                                        String[] examples_cpp = ["convex_hull","dot_product"]
                                        def test_pass_status = true
                                        for (String example : examples_dpcpp) {
                                            test_pass_status = test_pass_status && runExample(example,"-DCMAKE_CXX_COMPILER=dpcpp")
                                        }
                                        test_pass_status = test_pass_status && runExample("gamma_correction","-DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_CXX_FLAGS=-DBUILD_FOR_HOST")
                                        for (String example : examples_cpp) {
                                            test_pass_status = test_pass_status && runExample(example,"-DCMAKE_CXX_COMPILER=c++")
                                        }
                                        test_pass_status = test_pass_status && runExample("gamma_correction","-DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS=-DBUILD_FOR_HOST")

                                        if (test_pass_status != true) {
                                            echo "Some checks failed. Please check log to fix the issue."
                                            sh script: "exit -1", label: "Set failure"
                                        }
                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    fail_stage = fail_stage + "    " + "Check_Samples"
                                    echo "Exception is" + e.toString()
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        sh "exit -1"
                                    }
                                }
                            }
                        }
                    }
                }

            }

        }
    }

    post {
        always {
            script {
                if (user_in_github_group) {
                    if (build_ok) {
                        currentBuild.result = "SUCCESS"
                        githubStatus.setSuccess(this, pipeline_name)
                    } else {
                        currentBuild.result = "FAILURE"
                        githubStatus.setFailed(this, pipeline_name)
                    }
                }
            }
        }
    }

}

