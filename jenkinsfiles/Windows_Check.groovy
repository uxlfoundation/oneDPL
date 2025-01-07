import groovy.xml.MarkupBuilder
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

def shell(String command, String label_string = "Bat Command") {
    return bat(returnStdout: true, script: "sh -x -c \"${command}\"", label: label_string).trim()
}

String pipeline_name = "Jenkins/Win_Check (dpcpp_gpu_cxx_17, cl_tbb_cxx_11)"
build_ok = true
fail_stage = ""
user_in_github_group = false
boolean code_changed

def runExample(String test_name, String compiler, String compile_options, List oneapi_env) {
    try {
        withEnv(oneapi_env) {
            bat script: "d: && cd ${env.WORKSPACE}\\src\\examples\\" + test_name + "\\src && \
            echo \"Build&Test command: " + compiler + " " + compile_options + "/W0 /nologo /D _UNICODE /D UNICODE /Zi /WX- /EHsc /Fetest.exe /Isrc/include main.cpp -o test.exe && test.exe\"\
             && " + compiler + " " + compile_options + " /W0 /nologo /D _UNICODE /D UNICODE /Zi /WX- /EHsc /Fetest.exe /I${env.WORKSPACE}/src/include main.cpp -o test.exe && test.exe",
             label: test_name + " Test Step"
        }
    }
    catch(e) {
        build_ok = false
        fail_stage = fail_stage + "    " + "Check_Samples_" + name
        echo "Exception is" + e.toString()
        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
            bat 'exit 1'
        }
    }
}

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

        stage('Win_Check') {
            when {
                expression { user_in_github_group }
            }
            agent { label "oneDPL_Win" }
            stages {
                stage('Git-monorepo') {
                    steps {
                        script {
                            try {
                                retry(2) {
                                    deleteDir()

                                    bat script: """
                                        d:
                                        cd ${env.WORKSPACE}
                                        git clone https://github.com/uxlfoundation/oneDPL.git src
                                        cd src
                                        git config --local --add remote.origin.fetch +refs/pull/${env.PR_number}/head:refs/remotes/origin/pr/${env.PR_number}
                                        git pull origin
                                        git checkout ${env.Commit_id}
                                        git merge origin/${env.Base_branch}
                                     """
                                }
                            }
                            catch (e) {
                                build_ok = false
                                fail_stage = fail_stage + "    " + "Git-monorepo"
                                bat "exit -1"
                            }
                        }
                    }
                }

                stage('Check_code_changes') {
                    steps {
                        script {
                            dir("./src") {
                                code_changed = bat(
                                    script: "git diff --name-only main | findstr /V \"^documentation\"",
                                    returnStdout: true, label: "Code_changed")
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
                                bat script: """
                                        d:
                                        cd ${env.WORKSPACE}
                                        call D:\\netbatch\\iusers\\oneDPL_CI\\get_oneAPI_package.bat ${env.OneAPI_Package_Date}
                                     """

                                bat script: """
                                        d:
                                        cd ${env.WORKSPACE}
                                        call D:\\netbatch\\iusers\\oneDPL_CI\\setup_env.bat ${env.OneAPI_Package_Date}
                                        wcontext && call ${env.WORKSPACE}\\win_prod\\compiler\\env\\vars.bat && call ${env.WORKSPACE}\\win_prod\\dpcpp-ct\\env\\vars.bat && set>envs_tobe_loaded.txt
                                     """
                                oneapi_env = readFile('envs_tobe_loaded.txt').split('\r\n') as List
                            }
                            catch (e) {
                                build_ok = false
                                fail_stage = fail_stage + "    " + "Setting_Env"
                                bat "exit -1"
                            }
                        }
                    }
                }

                stage('Check_Samples') {
                    when {
                        expression { code_changed }
                    }
                    steps {
                        timeout(time: 1, unit: 'HOURS') {
                            script {
                                try {
                                    String[] examples_dpcpp = ["gamma_correction","stable_sort_by_key","convex_hull","dot_product","histogram","random"]
                                    String[] examples_cpp = ["convex_hull","dot_product"]
                                    for (String example : examples_dpcpp) {
                                        runExample(example, "dpcpp", "", oneapi_env)
                                    }
                                    runExample("gamma_correction", "dpcpp", "-DBUILD_FOR_HOST", oneapi_env)
                                    for (String example : examples_cpp) {
                                        runExample(example, "cl", "", oneapi_env)
                                    }
                                    runExample("gamma_correction", "cl", "-DBUILD_FOR_HOST", oneapi_env)
                                }
                                catch(e) {
                                    build_ok = false
                                    fail_stage = fail_stage + "    " + "Check_Samples"
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        bat 'exit 1'
                                    }
                                }
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
                                    script {
                                        withEnv(oneapi_env) {
                                            dir("./src/build") {
                                                bat script: """
                                                    set MAKE_PROGRAM=%DevEnvDir%CommonExtensions\\Microsoft\\CMake\\Ninja\\ninja.exe
                                                    rd /s /q . 2>nul
                                                    dpcpp --version
                                                    cmake -G "Ninja" -DCMAKE_MAKE_PROGRAM="%MAKE_PROGRAM%"^
                                                        -DCMAKE_TOOLCHAIN_FILE=cmake\\windows-dpcpp-toolchain.cmake^
                                                        -DCMAKE_CXX_STANDARD=17^
                                                        -DCMAKE_BUILD_TYPE=release^
                                                        -DCMAKE_CXX_COMPILER=dpcpp-cl^
                                                        -DONEDPL_BACKEND=dpcpp^
                                                        -DONEDPL_DEVICE_TYPE=GPU ..
                                                """, label: "Generate"
                                                bat script: """
                                                    set MAKE_PROGRAM=%DevEnvDir%CommonExtensions\\Microsoft\\CMake\\Ninja\\ninja.exe
                                                    "%MAKE_PROGRAM%" build-onedpl-tests -v -k 0
                                                """, label: "Build"
                                                bat script: """
                                                    ctest --output-on-failure -C release --timeout %TEST_TIMEOUT%
                                                """, label: "All tests"
                                            }
                                        }
                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    echo "Exception is" + e.toString()
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        bat 'exit 1'
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Tests_cl_tbb_cxx_11') {
                    when {
                        expression { code_changed }
                    }
                    steps {
                        timeout(time: 2, unit: 'HOURS') {
                            script {
                                try {
                                    script {
                                        withEnv(oneapi_env) {
                                            dir("./src/build") {
                                                bat script: """
                                                    set MAKE_PROGRAM=%DevEnvDir%CommonExtensions\\Microsoft\\CMake\\Ninja\\ninja.exe
                                                    rd /s /q . 2>nul
                                                    cmake -G "Ninja" -DCMAKE_MAKE_PROGRAM="%MAKE_PROGRAM%"^
                                                        -DCMAKE_CXX_STANDARD=11^
                                                        -DCMAKE_BUILD_TYPE=release^
                                                        -DCMAKE_CXX_COMPILER=cl^
                                                        -DONEDPL_BACKEND=tbb^
                                                        -DONEDPL_DEVICE_TYPE=HOST ..
                                                """, label: "Generate"
                                                bat script: """
                                                    set MAKE_PROGRAM=%DevEnvDir%CommonExtensions\\Microsoft\\CMake\\Ninja\\ninja.exe
                                                    "%MAKE_PROGRAM%" build-onedpl-tests -v -k 0
                                                """, label: "Build"
                                                bat script: """
                                                    ctest --output-on-failure -C release --timeout %TEST_TIMEOUT%
                                                """, label: "All tests"
                                            }
                                        }
                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    echo "Exception is" + e.toString()
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        bat 'exit 1'
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

