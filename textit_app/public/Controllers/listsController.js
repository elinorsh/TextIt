/**
 * Created by user on 26-Jun-17.
 */

app.controller('listsController', ['$http','$location', '$window', function($http,$location, $window) {
    let self = this;
    self.sentencesCommon = [];
    self.sentencesTwitter = [];

    self.initList = function (){
        $http.get('http://localhost:4000/get_common_list')
            .then(function (res) {
                self.sentencesCommon = res.data;
            });
        $http.get('http://localhost:4000/get_twitter_list')
            .then(function (res) {
                self.sentencesTwitter = res.data;
            });
    };



}]);
